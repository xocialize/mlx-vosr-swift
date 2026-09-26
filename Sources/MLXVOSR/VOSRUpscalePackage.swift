import Foundation
import CoreGraphics
import CoreImage
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import VOSRMLX

/// Errors at the VOSR package boundary.
public enum VOSRPackageError: Error, Equatable {
    case imageDecodeFailed(String)
    case imageEncodeFailed
    case weightsDirectoryUnresolved
}

/// An MLXEngine `imageUpscale` package over **VOSR2** (CSWRY/VOSR, CVPR 2026): a one-step generative
/// super-resolution model — 1.39 B LightningDiT conditioned on DINOv2-L features in the Qwen-Image 2-D VAE
/// latent space — registered as the **generative stills tier** beside Real-ESRGAN (fast), RealPLKSR
/// (web-photo) and SeedVR2-3B (diffusion). Ported directly from the reference (PORTING-SPEC: e2e 115.5 dB
/// vs the PyTorch CPU-fp32 oracle at ×1, tiled 121.8 dB; fp16 tier 49.3 dB, int8 46.7 dB).
///
/// Pipeline per request: decode → Pillow-exact bicubic ×scale (the reference's own input path) →
/// reflect-pad to a multiple of 16 → VAE encode → 512² latent-tiled DiT (per-tile DINOv2 conditioning,
/// Gaussian-blended velocities, one Euler step) → VAE decode → crop → encode. ⚠️ Generative: it rewrites
/// letters in already-legible small text (VOSR-EVAL AB-L-0172) — hosts route text-heavy inputs through
/// MaskKit's `TextLegibilityGuard` (ForgeCore `EngineTextGuardedUpscaler`).
///
/// Born sweep-clean (1.14): split footprint per quant, `QuantConfigured`, `BudgetAware` fp16→int8,
/// `unload()` flushes the MLX pool; born materialization-clean (`WeightSourcing`, engine-executed) and
/// cancel-clean (entry checkpoint + one per DiT tile, `RunProgress` per tile).
@InferenceActor
public final class VOSRUpscalePackage: ModelPackage {
    public typealias Configuration = VOSRConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: VOSR2 weights Apache-2.0 (CSWRY/VOSR; VAE = Qwen-Image / Wan Apache-2.0; DINOv2 Apache-2.0).
            // C8: the port is an isomorphic derivative of the upstream Apache-2.0 arch files → Apache-2.0.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(sourceRepo: VOSRConfiguration.fp16Repo, revision: "main", tier: 2),
            requirements: RequirementsManifest(
                // Split footprint (engine 1.14). resident = the loaded weight set; peakActivation = the
                // tile-bound DiT transient + the FULL-IMAGE VAE encode/decode at the output resolution
                // (the dominant term — it grows with output pixels until `vaeWholeFramePixels` tiles it).
                // Measured with `vosr-smoke run` (MLX peak, release, M5 Max, 2026-09-26; one process per row):
                //
                //   input → output        tier   run     MLX peak
                //   512²  → 512²  (×1)    fp16   0.3 s   4.67 GB      int8   0.3 s   3.32 GB
                //   512²  → 1024² (×2)    fp16   1.6 s   5.98 GB      int8   1.7 s   4.70 GB
                //   512²  → 2048² (×4)    fp16   5.7 s  13.90 GB      int8   6.7 s  12.62 GB   ← pre-chunking; see S7
                //   above 2048² of output the VAE runs in 1024-px Gaussian-blended tiles (`vaeWholeFramePixels`);
                //   a 4096² full-image VAE is impossible (the mid-block attention would need a 137 GB score matrix).
                //
                // ⚠️ PROVISIONAL — MLX-peak basis, not the in-app `phys_footprint` the governor compares
                // against (the sibling measured ~2.4× between the two). RE-BASELINE with the image-fleet
                // batch (AB-T-0019) and replace both numbers.
                footprints: [
                    QuantFootprint(quant: .fp16, residentBytes: 3_400_000_000, peakActivationBytes: 6_000_000_000),
                    QuantFootprint(quant: .int8, residentBytes: 2_000_000_000, peakActivationBytes: 6_000_000_000),
                    QuantFootprint(quant: .fp32, residentBytes: 6_700_000_000, peakActivationBytes: 7_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                ImageUpscaleContract.descriptor(
                    name: "vosr2-upscale",
                    summary: "VOSR2 one-step generative super-resolution (1.4B DiT + DINOv2-L conditioning), ×1–×4, 512² latent-tiled; the generative stills tier."
                )
            ]
        )
    }

    private let configuration: Configuration
    private var pipeline: VOSR2Pipeline?
    private var loadedQuant: Quant?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    /// The tier `load()` will use: the configured lane, or int8 when the stamped budget is tight (BudgetAware).
    public nonisolated var plannedQuant: Quant {
        var quant = configuration.effectiveQuant
        if quant == .fp16, let budget = configuration.availableBudgetBytes, budget < VOSRConfiguration.fp16MinBudgetBytes {
            quant = .int8
        }
        return quant
    }

    public func load() async throws {
        guard pipeline == nil else { return }   // idempotent (C13: re-loadable after unload)
        guard let dir = configuration.resolvedWeightsDirectory(storeRoot: configuration.modelsRootDirectory) else {
            throw VOSRPackageError.weightsDirectoryUnresolved
        }
        let quant = plannedQuant
        // Contract 1.24: the engine has already materialized the lane into `dir`; this just loads.
        let files: VOSR2Pipeline.WeightFiles = quant == .fp32 ? .fp32 : .fp16
        let p = try VOSR2Pipeline.load(from: dir, files: files, quantizeDiT: quant == .int8)
        p.tileSize = configuration.tileSize
        p.tileOverlap = configuration.tileOverlap
        pipeline = p
        loadedQuant = quant
    }

    public func unload() async {
        pipeline = nil
        loadedQuant = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation.
        try Task.checkCancellation()
        guard let pipeline else { throw PackageError.notLoaded }
        guard request.capability == .imageUpscale, let req = request as? ImageUpscaleRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        let src = try Self.decodeToRGB8(req.image)
        let scale = min(max(req.scale ?? configuration.defaultScale, 1), VOSRConfiguration.maxScale)
        let (outW, outH) = (src.width * scale, src.height * scale)

        // The reference's input path: Pillow BICUBIC ×scale of the 8-bit image (bit-exact port).
        var rgb = src.pixels
        if scale > 1 {
            rgb = PILBicubic.resize(rgb, width: src.width, height: src.height, channels: 3, toWidth: outW, toHeight: outH)
        }
        try Task.checkCancellation()
        // The VAE needs multiples of 8 and the patch grid multiples of 16: reflect-pad, crop after.
        let (padW, padH) = ((outW + 15) / 16 * 16, (outH + 15) / 16 * 16)
        if padW != outW || padH != outH {
            rgb = Self.reflectPad(rgb, width: outW, height: outH, channels: 3, toWidth: padW, toHeight: padH)
        }
        let x = Self.toTensor(rgb, width: padW, height: padH)

        pipeline.vaeTileSize = padW * padH > configuration.vaeWholeFramePixels ? configuration.vaeTileSize : nil
        // CAN-2: one cooperative checkpoint per DiT tile, rethrown unchanged; progress per tile (1.18).
        pipeline.checkpoint = { try Task.checkCancellation() }
        pipeline.onTile = { i, n in RunProgress.report(.upsample, step: i, totalSteps: n) }
        defer { pipeline.checkpoint = nil; pipeline.onTile = nil }

        let y = try pipeline.superResolve(x, seed: configuration.seed)
        let out = Self.toRGB8(y, cropWidth: outW, cropHeight: outH)

        // Output mirrors the input format: rawBGRA8 in ⇒ rawBGRA8 out (no re-encode); else .png.
        if req.image.format == .rawBGRA8 {
            return ImageUpscaleResponse(image: Self.encodeRawBGRA8(out, width: outW, height: outH), appliedScale: scale)
        }
        guard let png = Self.encodePNG(out, width: outW, height: outH) else { throw VOSRPackageError.imageEncodeFailed }
        return ImageUpscaleResponse(image: Image(format: .png, data: png, width: outW, height: outH), appliedScale: scale)
    }

    // MARK: - pixels ↔ tensors

    /// NCHW float32 in [−1, 1] (torchvision ToTensor · 2 − 1).
    public nonisolated static func toTensor(_ rgb: [UInt8], width: Int, height: Int) -> MLXArray {
        let f = rgb.map { Float($0) / 255.0 * 2.0 - 1.0 }
        return MLXArray(f, [1, height, width, 3]).transposed(0, 3, 1, 2)
    }

    /// NCHW [−1, 1] → RGB8 (ToPILImage on `x·0.5 + 0.5`: clamp, ×255, round-half-even), cropped to the requested size.
    public nonisolated static func toRGB8(_ x: MLXArray, cropWidth w: Int, cropHeight h: Int) -> [UInt8] {
        let v = clip(x[0, 0..., 0..<h, 0..<w].transposed(1, 2, 0) * 0.5 + 0.5, min: 0.0, max: 1.0) * 255.0
        return v.asType(.float32).asArray(Float.self).map { UInt8(min(255, max(0, Int($0.rounded(.toNearestOrEven))))) }
    }

    /// PyTorch `reflect` padding (edge excluded) on the right and bottom of an interleaved 8-bit image.
    public nonisolated static func reflectPad(_ src: [UInt8], width w: Int, height h: Int, channels c: Int,
                                       toWidth W: Int, toHeight H: Int) -> [UInt8] {
        precondition(W - w < w && H - h < h, "reflect pad needs pad < size")
        var out = [UInt8](repeating: 0, count: W * H * c)
        for y in 0..<H {
            let sy = y < h ? y : 2 * (h - 1) - y
            for x in 0..<W {
                let sx = x < w ? x : 2 * (w - 1) - x
                let s = (sy * w + sx) * c, d = (y * W + x) * c
                for k in 0..<c { out[d + k] = src[s + k] }
            }
        }
        return out
    }

    // MARK: - Image codec (the canonical Image ↔ BGRA seam, as the Real-ESRGAN / RealPLKSR siblings)

    /// Decode a canonical `Image` (.png/.jpeg/.rawBGRA8) to interleaved sRGB RGB8.
    public nonisolated static func decodeToRGB8(_ image: Image) throws -> (pixels: [UInt8], width: Int, height: Int) {
        if image.format == .rawBGRA8 {
            guard let w = image.width, let h = image.height, w > 0, h > 0 else {
                throw VOSRPackageError.imageDecodeFailed("rawBGRA8 requires width/height")
            }
            let stride = image.bytesPerRow ?? (w * 4)
            guard stride >= w * 4, image.data.count >= stride * h else {
                throw VOSRPackageError.imageDecodeFailed("rawBGRA8 data too small (\(image.data.count) < \(stride * h))")
            }
            var rgb = [UInt8](repeating: 0, count: w * h * 3)
            image.data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
                for y in 0..<h {
                    for x in 0..<w {
                        let s = y * stride + x * 4, d = (y * w + x) * 3
                        rgb[d] = src[s + 2]; rgb[d + 1] = src[s + 1]; rgb[d + 2] = src[s]
                    }
                }
            }
            return (rgb, w, h)
        }
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw VOSRPackageError.imageDecodeFailed("unreadable \(image.format.rawValue) data")
        }
        let (w, h) = (cg.width, cg.height)
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
        else { throw VOSRPackageError.imageDecodeFailed("CGContext for RGBA draw") }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) { rgb[i * 3] = rgba[i * 4]; rgb[i * 3 + 1] = rgba[i * 4 + 1]; rgb[i * 3 + 2] = rgba[i * 4 + 2] }
        return (rgb, w, h)
    }

    /// Encode interleaved RGB8 as PNG bytes (sRGB, no alpha).
    public nonisolated static func encodePNG(_ rgb: [UInt8], width w: Int, height h: Int) -> Data? {
        var rgba = [UInt8](repeating: 255, count: w * h * 4)
        for i in 0..<(w * h) { rgba[i * 4] = rgb[i * 3]; rgba[i * 4 + 1] = rgb[i * 3 + 1]; rgba[i * 4 + 2] = rgb[i * 3 + 2] }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let cg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                               space: CGColorSpace(name: CGColorSpace.sRGB)!,
                               bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue),
                               provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }

    /// Emit interleaved RGB8 as tightly-packed raw BGRA8 `Image` bytes (alpha 255).
    public nonisolated static func encodeRawBGRA8(_ rgb: [UInt8], width w: Int, height h: Int) -> Image {
        var out = Data(count: w * h * 4)
        out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            for i in 0..<(w * h) {
                dst[i * 4] = rgb[i * 3 + 2]; dst[i * 4 + 1] = rgb[i * 3 + 1]; dst[i * 4 + 2] = rgb[i * 3]; dst[i * 4 + 3] = 255
            }
        }
        return Image.rawBGRA8(data: out, width: w, height: h)
    }
}

extension VOSRUpscalePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(VOSRUpscalePackage.self)
    }
}
