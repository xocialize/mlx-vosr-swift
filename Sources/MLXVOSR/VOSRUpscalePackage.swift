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
///
/// **16 bits out** (contract 1.52.0; AB-T-0233, decision AB-D-0127 option (a)): the surface declares `.rawRGBA16`. A
/// `.rawRGBA16` image, or a PNG/TIFF deeper than 8 bits, is rounded to 8 bits for the reference prep (the Pillow bicubic
/// the oracle gate is defined on, unchanged), runs the same encode and DiT, and has its final latent decoded by a
/// float32 copy of the VAE at `VOSRConfiguration.deepDecodeTileSize` tiles. The model's float result comes back at 16
/// bits: `.rawRGBA16` for raw, a 16-bit PNG otherwise. 8-bit inputs take the 8-bit path byte for byte.
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
                // tile-bound DiT transient + the VAE encode/decode — whole-frame up to `vaeWholeFramePixels`
                // (1024² since AB-D-0120), 1024-px Gaussian-blended tiles above, so the peak no longer grows with
                // the output past 1024². RE-BASELINED 2026-10-10 to the in-app `phys_footprint` the governor compares
                // against (AB-T-0185 step 6; ForgeOptimizer `UpscaleMemoryBench` + `vosr-smoke engine`, M5 Max,
                // 512² still, one process per row):
                //
                //   input → output        tier   run      phys peak   MLX pool peak
                //   512²  → 1024² (×2)    fp16   5.3 s     9.1 GB     6.72 GB   (whole-frame VAE at 1024²)
                //   512²  → 2048² (×4)    fp16  28.9 s     9.6 GB     6.87 GB   (tiled VAE)
                //   512²  → 2048² (×4)    int8  28.3 s     8.4 GB     5.59 GB
                //   512²  → 2048² (×4)    fp16  28.8 s    20.1 GB    16.86 GB   ← the OLD whole-frame default; under the
                //                                                               engine's 2 GiB MLX cache cap that image
                //                                                               was WRONG before 0.2.2 (8.7 dB vs the
                //                                                               oracle; AB-D-0120, AB-T-0218)
                //
                // A 16-bit request (0.3.0, AB-T-0233) decodes with a float32 VAE at 768-px tiles: `vosr-smoke engine --deep`,
                // release, M5 Max, 2026-10-12, beside the same image at 8 bits:
                //   512²  → 2048² (×4)    fp16 + float32 decode   8.4 s   8.86 GB   6.10 GB   (8-bit: 8.66 GB)
                //   768×507 → 3072×2028   fp16 + float32 decode  12.8 s   9.00 GB   6.19 GB   (8-bit: 8.80 GB)
                // The +0.2 GB is the float32 VAE's weights (loaded on the first 16-bit request); both stay under the
                // declared 3.4 + 6.6 GB.
                //
                // Declared below with ~6 % margin (`VOSRConfiguration.*Footprint`); the fp32 lane is a derived bound.
                footprints: [
                    VOSRConfiguration.fp16Footprint,
                    VOSRConfiguration.int8Footprint,
                    VOSRConfiguration.fp32Footprint,
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
                // Mirrors rawBGRA8 (since 0.1.0) and writes 16 bits for rawRGBA16 and deep PNG (0.3.0).
                .declaring(rawImageFormats: [.rawBGRA8, .rawRGBA16])
            ]
        )
    }

    private let configuration: Configuration
    private var pipeline: VOSR2Pipeline?
    private var loadedQuant: Quant?
    /// Where `load()` read the weights, for the deep decoder's VAE.
    private var weightsDirectory: URL?
    /// The float32 VAE a 16-bit request decodes with: loaded on the first one (the fp16 VAE's weights cast, ~180 MB),
    /// the lane's own VAE on the fp32 lane. Dropped by `unload()`.
    private var deepDecoder: AutoencoderKLQwenImage2D?

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
        weightsDirectory = dir
    }

    public func unload() async {
        pipeline = nil
        loadedQuant = nil
        deepDecoder = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation.
        try Task.checkCancellation()
        guard let pipeline else { throw PackageError.notLoaded }
        guard request.capability == .imageUpscale, let req = request as? ImageUpscaleRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        // A 16-bit image is rounded to 8 bits for the reference prep (`deep` keeps the 16-bit decode and output).
        let deep = Self.isDeep(req.image)
        let src = deep ? try Self.decodeDeepToRGB8(req.image) : try Self.decodeToRGB8(req.image)
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

        let wholeFrame = Self.wholeFramePixels(configured: configuration.vaeWholeFramePixels,
                                               cacheLimit: Memory.cacheLimit, memoryLimit: Memory.memoryLimit)
        pipeline.vaeTileSize = padW * padH > wholeFrame ? configuration.vaeTileSize : nil
        // CAN-2: one cooperative checkpoint per DiT tile, rethrown unchanged; progress per tile (1.18).
        pipeline.checkpoint = { try Task.checkCancellation() }
        pipeline.onTile = { i, n in RunProgress.report(.upsample, step: i, totalSteps: n) }
        defer { pipeline.checkpoint = nil; pipeline.onTile = nil }

        let decoder = deep ? try deepDecoder(for: pipeline) : nil
        let y = try pipeline.superResolve(x, seed: configuration.seed, decoder: decoder)
        if deep {
            // 16 bits out: rawRGBA16 in ⇒ rawRGBA16 out, a deep PNG/TIFF in ⇒ a 16-bit PNG.
            let words = Self.toRGBA16(y, cropWidth: outW, cropHeight: outH)
            if req.image.format == .rawRGBA16 {
                return ImageUpscaleResponse(image: .rawRGBA16(data: Self.data(words), width: outW, height: outH), appliedScale: scale)
            }
            guard let png = Self.encodePNG16(words, width: outW, height: outH) else { throw VOSRPackageError.imageEncodeFailed }
            return ImageUpscaleResponse(image: Image(format: .png, data: png, width: outW, height: outH), appliedScale: scale)
        }
        let out = Self.toRGB8(y, cropWidth: outW, cropHeight: outH)

        // Output mirrors the input format: rawBGRA8 in ⇒ rawBGRA8 out (no re-encode); else .png.
        if req.image.format == .rawBGRA8 {
            return ImageUpscaleResponse(image: Self.encodeRawBGRA8(out, width: outW, height: outH), appliedScale: scale)
        }
        guard let png = Self.encodePNG(out, width: outW, height: outH) else { throw VOSRPackageError.imageEncodeFailed }
        return ImageUpscaleResponse(image: Image(format: .png, data: png, width: outW, height: outH), appliedScale: scale)
    }

    /// The decoder a 16-bit request uses: a float32 VAE (the lane's own on fp32; the fp16 VAE's weights cast otherwise,
    /// loaded once) at `VOSRConfiguration.deepDecodeTileSize` tiles.
    private func deepDecoder(for pipeline: VOSR2Pipeline) throws -> VOSR2Pipeline.Decoder {
        if deepDecoder == nil {
            if loadedQuant == .fp32 {
                deepDecoder = pipeline.vae
            } else {
                guard let dir = weightsDirectory,
                      let file = VOSRConfiguration.files(for: .fp16).first(where: { $0.hasPrefix("vae_") }) else {
                    throw VOSRPackageError.weightsDirectoryUnresolved
                }
                let vae = AutoencoderKLQwenImage2D()
                try vae.loadExact(url: dir.appendingPathComponent(file), component: "vae", cast: .float32)
                deepDecoder = vae
            }
        }
        return VOSR2Pipeline.Decoder(vae: deepDecoder!, tileSize: VOSRConfiguration.deepDecodeTileSize)
    }

    /// The largest output decoded whole-frame under a host's MLX cache cap: 1024², the envelope the declared footprints
    /// were measured on (AB-D-0120).
    public nonisolated static let cappedWholeFramePixels = 1024 * 1024

    /// The whole-frame VAE bound a run uses. A host that caps MLX's cache (the engine caps it at 2 GiB, AB-A-0130; MLX's
    /// own default cache limit is its memory limit) gets at most `cappedWholeFramePixels`. A whole-frame 2048² decode is
    /// correct under the cap since 0.2.2 (byte-sized attention chunks, AB-T-0218), but it needs a 16.9 GB MLX peak against
    /// a 6.6 GB declared activation, so an engine host never gets it. Without a cap (a raw host doing parity work) the
    /// configured value holds.
    public nonisolated static func wholeFramePixels(configured: Int, cacheLimit: Int, memoryLimit: Int) -> Int {
        cacheLimit < memoryLimit ? min(configured, cappedWholeFramePixels) : configured
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

    // MARK: - 16 bits (contract 1.52.0, AB-T-0233)

    /// RGBA, 16 bits per component, little-endian, premultiplied — `.rawRGBA16`'s layout as CoreGraphics draws it.
    public nonisolated static let rgba16BitmapInfo =
        CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder16Little.rawValue

    /// Whether `image` is deeper than 8 bits: `.rawRGBA16`, or a PNG/TIFF with more than 8 bits per component.
    public nonisolated static func isDeep(_ image: Image) -> Bool {
        if image.format == .rawRGBA16 { return true }
        guard image.format != .rawBGRA8, let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return false }
        return cg.bitsPerComponent > 8
    }

    /// A deep image's 16-bit RGBA words (sRGB), tightly packed.
    public nonisolated static func decodeRGBA16(_ image: Image) throws -> (words: [UInt16], width: Int, height: Int) {
        if image.format == .rawRGBA16 {
            guard let w = image.width, let h = image.height, w > 0, h > 0 else {
                throw VOSRPackageError.imageDecodeFailed("rawRGBA16 requires width/height")
            }
            let stride = image.bytesPerRow ?? (w * 8)
            guard stride >= w * 8, image.data.count >= stride * h else {
                throw VOSRPackageError.imageDecodeFailed("rawRGBA16 data too small (\(image.data.count) < \(stride * h))")
            }
            var words = [UInt16](repeating: 0, count: w * h * 4)
            words.withUnsafeMutableBytes { dst in
                image.data.withUnsafeBytes { src in
                    for row in 0..<h {
                        memcpy(dst.baseAddress!.advanced(by: row * w * 8), src.baseAddress!.advanced(by: row * stride), w * 8)
                    }
                }
            }
            return (words, w, h)
        }
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw VOSRPackageError.imageDecodeFailed("unreadable \(image.format.rawValue) data")
        }
        let (w, h) = (cg.width, cg.height)
        var words = [UInt16](repeating: 0, count: w * h * 4)
        let drawn: Bool = words.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 16, bytesPerRow: w * 8,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: rgba16BitmapInfo) else { return false }
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard drawn else { throw VOSRPackageError.imageDecodeFailed("CGContext for RGBA16 draw") }
        return (words, w, h)
    }

    /// A deep image as interleaved RGB8 for the reference prep: each component rounded, `(v·255 + 32767) / 65535`, the
    /// rounding Pixel Agency Desktop's 8-bit route sends (so a 16-bit request's DiT sees what that route's would).
    public nonisolated static func decodeDeepToRGB8(_ image: Image) throws -> (pixels: [UInt8], width: Int, height: Int) {
        let (words, w, h) = try decodeRGBA16(image)
        return (rgb8(rounding: words, count: w * h), w, h)
    }

    @inline(never)
    nonisolated static func rgb8(rounding words: [UInt16], count: Int) -> [UInt8] {
        var rgb = [UInt8](repeating: 0, count: count * 3)
        for p in 0..<count {
            for c in 0..<3 { rgb[p * 3 + c] = UInt8((UInt32(UInt16(littleEndian: words[p * 4 + c])) * 255 + 32767) / 65535) }
        }
        return rgb
    }

    /// NCHW [−1, 1] → RGBA16 words (`x·0.5 + 0.5`, clamped, ×65535 rounded; alpha opaque), cropped to the requested size.
    @inline(never)
    public nonisolated static func toRGBA16(_ x: MLXArray, cropWidth w: Int, cropHeight h: Int) -> [UInt16] {
        let v = clip(x[0, 0..., 0..<h, 0..<w].transposed(1, 2, 0) * 0.5 + 0.5, min: 0.0, max: 1.0).asType(.float32).asArray(Float.self)
        var words = [UInt16](repeating: UInt16.max, count: w * h * 4)
        for p in 0..<(w * h) {
            for c in 0..<3 { words[p * 4 + c] = UInt16((v[p * 3 + c] * 65535).rounded()).littleEndian }
        }
        return words
    }

    nonisolated static func data(_ words: [UInt16]) -> Data { words.withUnsafeBytes { Data($0) } }

    /// RGBA16 words as a 16-bit sRGB PNG.
    nonisolated static func encodePNG16(_ words: [UInt16], width w: Int, height h: Int) -> Data? {
        guard let provider = CGDataProvider(data: data(words) as CFData),
              let cg = CGImage(width: w, height: h, bitsPerComponent: 16, bitsPerPixel: 64, bytesPerRow: w * 8,
                               space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: rgba16BitmapInfo),
                               provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { return nil }
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
