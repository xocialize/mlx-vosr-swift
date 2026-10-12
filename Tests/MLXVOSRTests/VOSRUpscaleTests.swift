import Testing
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLXToolKit
@testable import MLXVOSR

/// Offline conformance checks — no Metal evaluation. Live upscaling is proven by `vosr-smoke engine`
/// (the real engine path) and the gates in PORTING-SPEC.md.
struct VOSRUpscaleTests {

    @Test func manifestIsImageUpscaleAndPermissiveOnBothLayers() {
        let m = VOSRUpscalePackage.manifest
        #expect(m.capabilities == [.imageUpscale])
        #expect(m.license.weightLicense == .apache2)      // C7 — VOSR2 / Qwen-Image VAE / DINOv2, all Apache-2.0
        #expect(m.license.portCodeLicense == .apache2)    // C8 — isomorphic derivative of the Apache-2.0 arch files
        #expect(LicensePolicy.permissiveOnly.evaluate(m.license) == .admitted)
    }

    @Test func provenancePointsAtANamespaceWeControl() {
        // Fleet policy (2026-08-03): every weight source names a namespace WE control — mlx-community (member
        // org, the `<UpstreamRepoName>-<quant>` grammar) or xocialize.
        let ok: (String) -> Bool = { $0.hasPrefix("mlx-community/") || $0.hasPrefix("xocialize/") }
        #expect(ok(VOSRUpscalePackage.manifest.provenance.sourceRepo))
        #expect(ok(VOSRConfiguration.fp16Repo) && ok(VOSRConfiguration.fp32Repo))
        #expect(VOSRConfiguration(quant: .int8).repo == VOSRConfiguration.fp16Repo)
        #expect(VOSRConfiguration(quant: .fp32).repo == VOSRConfiguration.fp32Repo)
    }

    @Test func manifestRequirements() {
        let r = VOSRUpscalePackage.manifest.requirements
        #expect(r.requiredBackends.contains(.metalGPU))
        #expect(r.os.minMacOS == SemanticVersion(major: 26, minor: 0, patch: 0))
    }

    /// Efficiency adoption (engine 1.14): every offered tier declares the split (weights floor + the
    /// output-resolution VAE transient), and int8 is the lighter resident of the two half tiers.
    @Test func splitFootprintsDeclaredPerTier() {
        let fps = VOSRUpscalePackage.manifest.requirements.footprints
        for q in [Quant.fp16, .int8, .fp32] {
            let fp = fps.first { $0.quant == q }
            #expect(fp != nil, "\(q) footprint missing")
            #expect((fp?.peakActivationBytes ?? 0) > 0 && (fp?.residentBytes ?? 0) > 0)
        }
        let fp16 = fps.first { $0.quant == .fp16 }!, int8 = fps.first { $0.quant == .int8 }!
        #expect(int8.residentBytes < fp16.residentBytes)
    }

    @Test func quantConfiguredAndBudgetAware() {
        let cfg: any PackageConfiguration = VOSRConfiguration()
        #expect((cfg as? QuantConfigured)?.quant == .fp16)
        #expect(cfg is BudgetAware && cfg is ModelStorable && cfg is WeightSourcing && cfg is WeightPrewarming)
        // bf16 is not an offered lane: it resolves to fp16.
        #expect(VOSRConfiguration(quant: .bf16).effectiveQuant == .fp16)
        #expect(VOSRConfiguration(quant: .int4).effectiveQuant == .fp16)
        // Under a tight budget the fp16 plan drops to int8; int8 and fp32 plans are untouched.
        #expect(VOSRUpscalePackage(configuration: VOSRConfiguration(quant: .fp16, availableBudgetBytes: 4_000_000_000)).plannedQuant == .int8)
        #expect(VOSRUpscalePackage(configuration: VOSRConfiguration(quant: .fp16, availableBudgetBytes: 12_000_000_000)).plannedQuant == .fp16)
        #expect(VOSRUpscalePackage(configuration: VOSRConfiguration(quant: .fp32, availableBudgetBytes: 4_000_000_000)).plannedQuant == .fp32)
    }

    @Test func surfaceIsTheCanonicalUpscaleDescriptor() {
        let s = VOSRUpscalePackage.manifest.surfaces.first
        #expect(s?.capability == .imageUpscale)
        #expect(s?.parameters.first?.kind == .image)
        #expect(s?.parameters.contains { $0.name == "scale" && !$0.required } == true)
    }

    @Test func registrationConstructs() throws {
        let reg = VOSRUpscalePackage.registration
        #expect(reg.manifest.capabilities == [.imageUpscale])
        let pkg = try reg.makePackage(VOSRConfiguration())
        #expect(pkg is VOSRUpscalePackage)
    }

    @Test func weightSourcesFollowTheLane() {
        let half = VOSRConfiguration(quant: .int8).weightSources
        #expect(half.count == 1 && half[0].role == "vosr2-fp16" && half[0].repo == VOSRConfiguration.fp16Repo)
        #expect(half[0].matching?.contains("transformer_fp16.safetensors") == true)
        #expect(half[0].matching?.contains("transformer_fp32.safetensors") == false)
        let full = VOSRConfiguration(quant: .fp32).weightSources
        #expect(full[0].role == "vosr2-fp32" && full[0].repo == VOSRConfiguration.fp32Repo
                && full[0].matching?.contains("transformer_fp32.safetensors") == true)
    }

    @Test func explicitWeightsDirectoryWinsOverTheStore() throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("vosr-cfg-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let cfg = VOSRConfiguration(weightsDirectory: tmp, modelsRootDirectory: URL(fileURLWithPath: "/nowhere"))
        #expect(cfg.resolvedWeightsDirectory(storeRoot: cfg.modelsRootDirectory) == tmp)
        // Incomplete explicit dir → the lane is reported missing (never a silent partial load).
        #expect(cfg.missingWeightSources(storeRoot: nil).count == 1)
        for f in VOSRConfiguration.files(for: .fp16) { FileManager.default.createFile(atPath: tmp.appendingPathComponent(f).path, contents: Data([0])) }
        #expect(cfg.missingWeightSources(storeRoot: nil).isEmpty)
        #expect(cfg.prewarmPaths.count == 4)
    }

    @Test func configurationCodableRoundTrips() throws {
        let c = VOSRConfiguration(quant: .int8, seed: 9, defaultScale: 4, tileSize: 512, tileOverlap: 32,
                                  weightsDirectory: URL(fileURLWithPath: "/x"), availableBudgetBytes: 1)
        let back = try JSONDecoder().decode(VOSRConfiguration.self, from: JSONEncoder().encode(c))
        #expect(back.quant == .int8 && back.seed == 9 && back.defaultScale == 4)
        #expect(back.weightsDirectory == nil && back.availableBudgetBytes == nil && back.modelsRootDirectory == nil)
    }

    @Test func pngDecodesToRGB8AndReencodes() throws {
        let png = try #require(Self.makePNG(width: 32, height: 20))
        let image = Image(format: .png, data: png, width: 32, height: 20)
        let rgb = try VOSRUpscalePackage.decodeToRGB8(image)
        #expect(rgb.width == 32 && rgb.height == 20 && rgb.pixels.count == 32 * 20 * 3)
        #expect(rgb.pixels[0] == 153 && rgb.pixels[1] == 153 && rgb.pixels[2] == 153)   // gray 0.6 sRGB
        let back = try #require(VOSRUpscalePackage.encodePNG(rgb.pixels, width: 32, height: 20))
        #expect(back.prefix(4) == Data([0x89, 0x50, 0x4E, 0x47]))
    }

    @Test func rawBGRA8RoundTripsBitIdentical() throws {
        let w = 8, h = 4
        var bytes = Data((0..<(w * h * 4)).map { UInt8($0 % 256) })
        for i in stride(from: 3, to: bytes.count, by: 4) { bytes[i] = 255 }   // alpha is always 255 on the way out
        let rgb = try VOSRUpscalePackage.decodeToRGB8(Image.rawBGRA8(data: bytes, width: w, height: h))
        let back = VOSRUpscalePackage.encodeRawBGRA8(rgb.pixels, width: w, height: h)
        #expect(back.format == .rawBGRA8 && back.width == w && back.height == h && back.bytesPerRow == nil)
        #expect(back.data == bytes)
    }

    @Test func rawBGRA8MissingDimensionsThrows() {
        #expect(throws: VOSRPackageError.self) {
            _ = try VOSRUpscalePackage.decodeToRGB8(Image(format: .rawBGRA8, data: Data(count: 16)))
        }
    }

    @Test func reflectPadMirrorsWithoutTheEdge() {
        // 4×1 row 10 20 30 40 padded to width 6 → 10 20 30 40 30 20 (torch 'reflect').
        let out = VOSRUpscalePackage.reflectPad([10, 20, 30, 40], width: 4, height: 1, channels: 1, toWidth: 6, toHeight: 1)
        #expect(out == [10, 20, 30, 40, 30, 20])
    }

    static func makePNG(width: Int, height: Int) -> Data? {
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.setFillColor(CGColor(colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!, components: [0.6, 0.6, 0.6, 1])!)
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        guard let cg = ctx.makeImage() else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }
}

/// Under a host's MLX cache cap the whole-frame VAE bound is clamped to the declared envelope (AB-T-0218); MLX's own default
/// cache limit equals its memory limit, which is no cap.
struct VOSRWholeFrameClampTests {
    let gib = 1 << 30

    @Test func cappedHostsDecodeAtMost1024SquaredWholeFrame() {
        #expect(VOSRUpscalePackage.wholeFramePixels(configured: 2048 * 2048, cacheLimit: 2 * gib, memoryLimit: 150 * gib) == 1024 * 1024)
        #expect(VOSRUpscalePackage.wholeFramePixels(configured: 1024 * 1024, cacheLimit: 2 * gib, memoryLimit: 150 * gib) == 1024 * 1024)
        #expect(VOSRUpscalePackage.wholeFramePixels(configured: 512 * 512, cacheLimit: 2 * gib, memoryLimit: 150 * gib) == 512 * 512)
    }

    @Test func uncappedHostsKeepTheConfiguredLever() {
        #expect(VOSRUpscalePackage.wholeFramePixels(configured: 2048 * 2048, cacheLimit: 150 * gib, memoryLimit: 150 * gib) == 2048 * 2048)
        #expect(VOSRConfiguration().vaeWholeFramePixels == VOSRUpscalePackage.cappedWholeFramePixels)
    }
}
