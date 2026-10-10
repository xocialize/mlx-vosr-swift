// vosr-smoke — the CLI gate lane for the VOSR2 port (PORTING-SPEC.md phases S0–S2b, S6).
//
//   vosr-smoke keys <weightsDir> [fp32|fp16]                       S0: key contract, 0 missing / 0 unused
//   vosr-smoke gate <weightsDir> <goldens.safetensors> [--only a,b] [--gpu]
//                                                                  S1/S2: per-component parity vs the PyTorch
//                                                                  CPU-fp32 oracle (vae, dino, dit, e2e, tiled;
//                                                                  quant = the bf16 / int8 tiers, --gpu only)
//   vosr-smoke run <in.png> <out.png> <weightsDir> [--scale N] [--fp16|--bf16] [--int8] [--seed N] [--cpu] [--vae-tile N]
//                                                                  S2b: full pipeline on a real image
//   vosr-smoke engine <in.png> <out.png> <weightsDir> [--scale N] [--int8|--fp32] [--raw]
//                                                                  S7: VOSRUpscalePackage through the REAL
//                                                                  MLXServeEngine (register → run), explicit dir
//
// Gates run on the CPU stream by default (the numerically strict lane; fp32 goldens). `--gpu` reruns them
// on Metal to expose the GPU-vs-CPU fp32 gap (TF32 GEMMs on M5, the Winograd conv window).

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXNN
import MLXToolKit
import MLXServeCore
import VOSRMLX
import MLXVOSR

enum SmokeError: Error, CustomStringConvertible {
    case usage(String), badImage(String), gateFailed([String])
    var description: String {
        switch self {
        case .usage(let s): return "usage: \(s)"
        case .badImage(let s): return "bad image: \(s)"
        case .gateFailed(let f): return "GATE FAILED: \(f.joined(separator: ", "))"
        }
    }
}

func log(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
func now() -> Double { Date().timeIntervalSince1970 }

// MARK: - comparison

struct Delta: CustomStringConvertible {
    let name: String
    let maxAbs: Float
    let meanAbs: Float
    let refMax: Float
    let cosine: Float
    let shapeOK: Bool
    var relMax: Float { refMax > 0 ? maxAbs / refMax : maxAbs }
    var description: String {
        String(format: "%-28@ max|Δ| %.3e  mean|Δ| %.3e  |ref|max %.3g  rel %.2e  cos %.7f%@",
               name as NSString, maxAbs, meanAbs, refMax, relMax, cosine, shapeOK ? "" : "  SHAPE MISMATCH")
    }
}

func compare(_ name: String, _ got: MLXArray, _ ref: MLXArray) -> Delta {
    let g = got.asType(.float32), r = ref.asType(.float32)
    guard g.shape == r.shape else {
        return Delta(name: name, maxAbs: .infinity, meanAbs: .infinity, refMax: 0, cosine: 0, shapeOK: false)
    }
    let d = abs(g - r)
    let num = sum(g * r).item(Float.self)
    let den = (sqrt(sum(g * g)) * sqrt(sum(r * r))).item(Float.self)
    return Delta(name: name, maxAbs: d.max().item(Float.self), meanAbs: d.mean().item(Float.self),
                 refMax: abs(r).max().item(Float.self), cosine: den > 0 ? num / den : 0, shapeOK: true)
}

/// PSNR in dB for signals in [−1, 1] (peak-to-peak 2).
func psnr(_ got: MLXArray, _ ref: MLXArray) -> Float {
    let mse = mean(square(got.asType(.float32) - ref.asType(.float32))).item(Float.self)
    return mse > 0 ? 10 * log10f(4.0 / mse) : .infinity
}

// MARK: - image I/O (CLI only; the core is MLX-only)

/// PNG → interleaved RGB8 (alpha dropped, sRGB bytes untouched).
func loadRGB8(_ path: String) throws -> (pixels: [UInt8], width: Int, height: Int) {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw SmokeError.badImage(path) }
    let (w, h) = (cg.width, cg.height)
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    guard let ctx = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: cs,
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
    else { throw SmokeError.badImage("context") }
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    var rgb = [UInt8](repeating: 0, count: w * h * 3)
    for i in 0..<(w * h) { rgb[i * 3] = rgba[i * 4]; rgb[i * 3 + 1] = rgba[i * 4 + 1]; rgb[i * 3 + 2] = rgba[i * 4 + 2] }
    return (rgb, w, h)
}

func savePNG(rgb: [UInt8], width: Int, height: Int, to path: String) throws {
    var rgba = [UInt8](repeating: 255, count: width * height * 4)
    for i in 0..<(width * height) { rgba[i * 4] = rgb[i * 3]; rgba[i * 4 + 1] = rgb[i * 3 + 1]; rgba[i * 4 + 2] = rgb[i * 3 + 2] }
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    guard let provider = CGDataProvider(data: Data(rgba) as CFData),
          let cg = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                           space: cs, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw SmokeError.badImage("encode") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw SmokeError.badImage("finalize") }
}

/// RGB8 → NCHW float32 in [−1, 1] (torchvision ToTensor * 2 − 1).
func toTensor(_ rgb: [UInt8], width: Int, height: Int) -> MLXArray {
    let f = rgb.map { Float($0) / 255.0 * 2.0 - 1.0 }
    return MLXArray(f, [1, height, width, 3]).transposed(0, 3, 1, 2)
}

/// NCHW [−1, 1] → RGB8 (torchvision ToPILImage on `x * 0.5 + 0.5`: clamp, × 255, round-half-even → uint8).
func toRGB8(_ x: MLXArray) -> (pixels: [UInt8], width: Int, height: Int) {
    let (h, w) = (x.dim(2), x.dim(3))
    let v = clip(x[0].transposed(1, 2, 0) * 0.5 + 0.5, min: 0.0, max: 1.0) * 255.0
    let arr = v.asType(.float32).asArray(Float.self)
    return (arr.map { UInt8(min(255, max(0, Int(($0).rounded(.toNearestOrEven))))) }, w, h)
}

// MARK: - modes

func runKeys(_ dir: URL, lane: String) throws {
    var failures: [String] = []
    for (name, file, make) in [("vae", "vae_\(lane).safetensors", { AutoencoderKLQwenImage2D() as Module }),
                               ("dinov2", "dinov2_\(lane).safetensors", { DINOv2ViT() as Module }),
                               ("transformer", "transformer_\(lane).safetensors", { LightningDiT() as Module })] {
        let t0 = now()
        let m = make()
        let params = m.parameterCount
        do {
            try m.loadExact(url: dir.appendingPathComponent(file), component: name)
            log(String(format: "S0 %-12@ %@  %d tensors, %.3fB params, loaded in %.1fs — 0 missing / 0 unused ✅",
                       name as NSString, file as NSString, m.parameters().flattened().count, Double(params) / 1e9, now() - t0))
        } catch {
            log("S0 \(name) ❌ \(error)"); failures.append(name)
        }
    }
    if !failures.isEmpty { throw SmokeError.gateFailed(failures) }
}

func runGate(_ dir: URL, goldens gURL: URL, only: Set<String>, gpu: Bool) throws {
    let G = try MLX.loadArrays(url: gURL)
    func g(_ k: String) -> MLXArray { G[k]! }
    var report: [Delta] = []
    var failures: [String] = []
    func check(_ d: Delta, relTol: Float) {
        report.append(d); log("  " + d.description + (d.relMax <= relTol && d.shapeOK ? "  ✅" : "  ❌ (tol \(relTol))"))
        if !(d.relMax <= relTol && d.shapeOK) { failures.append(d.name) }
    }
    let lane = gpu ? "GPU" : "CPU"
    log("gate lane: \(lane) fp32 · goldens \(gURL.lastPathComponent) (\(G.count) arrays)")
    let want = { (s: String) in only.isEmpty || only.contains(s) }

    var vae: AutoencoderKLQwenImage2D? = nil
    var venc: DINOv2ViT? = nil
    var dit: LightningDiT? = nil
    if want("vae") || want("e2e") || want("tiled") {
        let t0 = now(); vae = AutoencoderKLQwenImage2D()
        try vae!.loadExact(url: dir.appendingPathComponent("vae_fp32.safetensors"), component: "vae")
        log(String(format: "loaded vae in %.1fs", now() - t0))
    }
    if want("dino") || want("e2e") || want("tiled") {
        let t0 = now(); venc = DINOv2ViT()
        try venc!.loadExact(url: dir.appendingPathComponent("dinov2_fp32.safetensors"), component: "dinov2")
        log(String(format: "loaded dinov2 in %.1fs", now() - t0))
    }
    if want("dit") || want("e2e") || want("tiled") {
        let t0 = now(); dit = LightningDiT()
        try dit!.loadExact(url: dir.appendingPathComponent("transformer_fp32.safetensors"), component: "transformer")
        log(String(format: "loaded transformer in %.1fs", now() - t0))
    }

    if want("vae"), let vae {
        log("[S1a] VAE")
        let t0 = now()
        let input = g("input")
        let moments = vae.moments(input); eval(moments)
        check(compare("vae.quant_conv", moments, g("vae.quant_conv")), relTol: 2e-3)
        let latentRaw = vae.encodeMode(input); eval(latentRaw)
        check(compare("vae.latent_raw", latentRaw, g("vae.latent_raw")), relTol: 2e-3)
        let latentNorm = vae.encodeNormalized(input); eval(latentNorm)
        check(compare("vae.latent_norm", latentNorm, g("vae.latent_norm")), relTol: 2e-3)
        let dec = vae.decodeNormalized(g("vae.latent_norm")); eval(dec)
        check(compare("vae.decode_of_input_latent", dec, g("vae.decode_of_input_latent")), relTol: 5e-3)
        log(String(format: "  vae PSNR(decode) %.1f dB · %.1fs", psnr(dec, g("vae.decode_of_input_latent")), now() - t0))
    }
    if want("dino"), let venc {
        log("[S1b] DINOv2")
        let t0 = now()
        let lq = g("input")
        let pre = venc.preprocess(0.5 * lq + 0.5); eval(pre)
        check(compare("dino.input448", pre, g("dino.input448")), relTol: 1e-4)
        let tokens = venc.prepareTokens(g("dino.input448"))
        let pe = venc.patchEmbed(g("dino.input448").transposed(0, 2, 3, 1)); eval(pe)
        check(compare("dino.patch_embed", pe, g("dino.patch_embed")), relTol: 1e-3)
        let b0 = venc.blocks[0](tokens); eval(b0)
        check(compare("dino.block0", b0, g("dino.block0")), relTol: 1e-3)
        let h17 = venc.hiddenStates(g("dino.input448"), upTo: 17); eval(h17)
        check(compare("dino.block17", h17, g("dino.block17")), relTol: 2e-3)
        let z = venc.features(g("dino.input448")); eval(z)
        check(compare("dino.z", z, g("dino.z")), relTol: 2e-3)
        log(String(format: "  %.1fs", now() - t0))
    }
    if want("dit"), let dit {
        log("[S1c] LightningDiT sub-ops + full forward")
        let t0 = now()
        let inp = g("dit.input")
        let xe = dit.xEmbedder(inp.transposed(0, 2, 3, 1)); eval(xe)
        check(compare("dit.x_embedder", xe, g("dit.x_embedder")), relTol: 1e-3)
        let t1 = MLXArray([Float(1.0)])
        let te = dit.tEmbedder(t1); eval(te)
        check(compare("dit.t_embedder", te, g("dit.t_embedder")), relTol: 1e-3)
        let tb = dit.tBlock(te); eval(tb)
        check(compare("dit.t_block", tb, g("dit.t_block")), relTol: 1e-3)
        let ln = dit.layerNorm(g("dino.z")); eval(ln)
        check(compare("dit.layer_norm", ln, g("dit.layer_norm")), relTol: 1e-3)
        let zz = dit.mlpCa(ln); eval(zz)
        check(compare("dit.mlp_ca", zz, g("dit.mlp_ca")), relTol: 1e-3)
        // block 0 internals from the golden inputs
        let (cos, sin) = dit.featRope.tables(for: 32, dtype: .float32)
        let x0 = g("dit.x_embedder"), c0 = g("dit.t_block"), zg = g("dit.mlp_ca")
        let n1 = dit.blocks[0].norm1(x0); eval(n1)
        check(compare("dit.block0.norm1", n1, g("dit.block0.norm1")), relTol: 1e-3)
        let b0 = dit.blocks[0](x0, c0: c0, z: zg, rope: (cos, sin)); eval(b0)
        check(compare("dit.block0", b0, g("dit.block0")), relTol: 1e-3)
        let b1 = dit.blocks[1](g("dit.block0"), c0: c0, z: zg, rope: (cos, sin)); eval(b1)
        check(compare("dit.block1", b1, g("dit.block1")), relTol: 1e-3)
        // chained through the stack
        var h = x0
        for i in 0..<dit.blocks.count {
            h = dit.blocks[i](h, c0: c0, z: zg, rope: (cos, sin))
            if i == 17 || i == 35 { eval(h); check(compare("dit.block\(i) (chained)", h, g("dit.block\(i)")), relTol: 5e-3) }
        }
        let fl = dit.finalLayer(g("dit.block35"), c: te); eval(fl)
        check(compare("dit.final_layer", fl, g("dit.final_layer")), relTol: 2e-3)
        let u = dit(inp, t: t1, z: g("dino.z")); eval(u)
        check(compare("dit.output", u, g("dit.output")), relTol: 5e-3)
        log(String(format: "  %.1fs", now() - t0))
    }
    if want("e2e"), let vae, let venc, let dit {
        log("[S2] end-to-end 512² (torch noise injected)")
        let t0 = now()
        let pipe = VOSR2Pipeline(vae: vae, venc: venc, model: dit)
        let (out, trace) = try pipe.superResolveTraced(g("input"), noise: g("dit.noise"))
        check(compare("e2e.latent_norm", trace.latent, g("vae.latent_norm")), relTol: 2e-3)
        check(compare("e2e.velocity", trace.velocity, g("dit.output")), relTol: 5e-3)
        check(compare("e2e.latent_out", trace.latentOut, g("dit.latent_out")), relTol: 5e-3)
        check(compare("e2e.output", out, g("output")), relTol: 1e-2)
        log(String(format: "  e2e PSNR %.1f dB vs oracle · %.1fs", psnr(out, g("output")), now() - t0))
    }
    if want("tiled"), let vae, let venc, let dit {
        log("[S5] tiled 1024² (3×3 latent tiles, torch noise injected)")
        let t0 = now()
        let pipe = VOSR2Pipeline(vae: vae, venc: venc, model: dit)
        pipe.onTile = { i, n in if i == 1 || i == n { log("   tile \(i)/\(n)") } }
        let (out, trace) = try pipe.superResolveTraced(g("tiled.input"), noise: g("tiled.noise"))
        check(compare("tiled.latent_norm", trace.latent, g("tiled.latent_norm")), relTol: 2e-3)
        check(compare("tiled.output", out, g("tiled.output")), relTol: 1e-2)
        log(String(format: "  tiled PSNR %.1f dB vs oracle · %.1fs", psnr(out, g("tiled.output")), now() - t0))
    }
    if want("quant") {
        guard gpu else { log("[S6] quant gate needs --gpu (quantized matmul is Metal-only)"); failures.append("quant(--gpu)"); return }
        log("[S6] half-precision and int8 DiT tiers vs the fp32 oracle (torch noise injected; VAE/DINOv2 fp16)")
        let vaeQ = AutoencoderKLQwenImage2D()
        try vaeQ.loadExact(url: dir.appendingPathComponent("vae_fp16.safetensors"), component: "vae")
        let vencQ = DINOv2ViT()
        try vencQ.loadExact(url: dir.appendingPathComponent("dinov2_fp16.safetensors"), component: "dinov2")
        for (tag, file, int8) in [("fp16", "transformer_fp16.safetensors", false), ("fp16+int8", "transformer_fp16.safetensors", true),
                                  ("bf16", "transformer_bf16.safetensors", false)] {
            let t0 = now()
            let ditQ = LightningDiT()
            try ditQ.loadExact(url: dir.appendingPathComponent(file), component: "transformer")
            if int8 { ditQ.quantizeLinears(groupSize: 64, bits: 8) }
            let pipe = VOSR2Pipeline(vae: vaeQ, venc: vencQ, model: ditQ)
            let (out, trace) = try pipe.superResolveTraced(g("input"), noise: g("dit.noise"))
            let dv = compare("\(tag).velocity", trace.velocity, g("dit.output")); report.append(dv)
            let dz = compare("\(tag).latent_out", trace.latentOut, g("dit.latent_out")); report.append(dz)
            let dout = compare("\(tag).output", out, g("output")); report.append(dout)
            let ok = dv.cosine >= 0.999 && dout.cosine >= 0.999
            log("  " + dv.description + (ok ? "  ✅" : "  ❌"))
            log("  " + dz.description)
            log("  " + dout.description)
            log(String(format: "  %@ PSNR %.1f dB vs fp32 oracle · %.1fs · peak MLX %.2f GB", tag as NSString, psnr(out, g("output")), now() - t0, Double(Memory.peakMemory) / 1e9))
            if !ok { failures.append("\(tag) (cos < 0.999)") }
        }
    }
    log("---- \(lane) gate summary: \(report.count - failures.count)/\(report.count) pass")
    if !failures.isEmpty { throw SmokeError.gateFailed(failures) }
}

func runImage(_ inPath: String, _ outPath: String, _ dir: URL, scale: Int, dtype: String, int8: Bool, seed: UInt64, vaeTile: Int?) throws {
    let t0 = now()
    let files: VOSR2Pipeline.WeightFiles = dtype == "bf16" ? .bf16 : (dtype == "fp16" ? .fp16 : .fp32)
    let pipe = try VOSR2Pipeline.load(from: dir, files: files, quantizeDiT: int8)
    pipe.vaeTileSize = vaeTile
    log(String(format: "loaded pipeline (%@%@) in %.1fs", dtype as NSString, int8 ? " + int8 DiT" : "", now() - t0))
    let img = try loadRGB8(inPath)
    var (rgb, w, h) = (img.pixels, img.width, img.height)
    if scale > 1 {
        let t1 = now()
        rgb = PILBicubic.resize(rgb, width: w, height: h, channels: 3, toWidth: w * scale, toHeight: h * scale)
        w *= scale; h *= scale
        log(String(format: "PIL-bicubic ×%d → %d×%d in %.2fs", scale, w, h, now() - t1))
    }
    guard w % 16 == 0, h % 16 == 0 else {
        throw SmokeError.badImage("dimensions after upsample must be multiples of 16 (got \(w)×\(h)); pad first")
    }
    let x = toTensor(rgb, width: w, height: h)
    pipe.onTile = { i, n in log("  tile \(i)/\(n)") }
    let t2 = now()
    let y = try pipe.superResolve(x, seed: seed)
    log(String(format: "superResolve %d×%d in %.1fs (peak MLX %.2f GB)", w, h, now() - t2, Double(Memory.peakMemory) / 1e9))
    let out = toRGB8(y)
    try savePNG(rgb: out.pixels, width: out.width, height: out.height, to: outPath)
    var lum = 0.0; for i in stride(from: 0, to: out.pixels.count, by: 3) { lum += Double(out.pixels[i]) * 0.299 + Double(out.pixels[i + 1]) * 0.587 + Double(out.pixels[i + 2]) * 0.114 }
    log(String(format: "wrote %@ · mean luma %.1f", outPath as NSString, lum / Double(out.pixels.count / 3)))
}

func runEngine(_ inPath: String, _ outPath: String, _ dir: URL, scale: Int?, quant: Quant, raw: Bool, vaeWholeFramePixels: Int? = nil) async throws {
    let img = try loadRGB8(inPath)
    let inputImage: MLXToolKit.Image
    if raw {
        inputImage = VOSRUpscalePackage.encodeRawBGRA8(img.pixels, width: img.width, height: img.height)
    } else {
        inputImage = MLXToolKit.Image(format: .png, data: try Data(contentsOf: URL(fileURLWithPath: inPath)), width: img.width, height: img.height)
    }
    var config = VOSRConfiguration(quant: quant, weightsDirectory: dir)
    if let px = vaeWholeFramePixels { config.vaeWholeFramePixels = px }
    // Full engine path: register (license gate + C10 eligibility) → run (engine constructs/loads the
    // package, C13; explicit weightsDirectory ⇒ nothing to materialize).
    let engine = MLXServeEngine()
    await engine.useModelStore(ModelStore(root: nil))
    let t0 = now()
    let id = try await engine.register(VOSRUpscalePackage.registration, configuration: config)
    let tReg = now() - t0
    var tiles: [String] = []
    let t1 = now()
    let resp = try await RunProgress.$sink.withValue({ r in tiles.append("\(r.step ?? 0)/\(r.totalSteps ?? 0)") }) {
        try await engine.run(ImageUpscaleRequest(image: inputImage, scale: scale), package: id)
    }
    let tRun = now() - t1
    guard let up = resp as? ImageUpscaleResponse else { throw SmokeError.badImage("unexpected response") }
    let out: (pixels: [UInt8], width: Int, height: Int)
    if up.image.format == .rawBGRA8 {
        out = try VOSRUpscalePackage.decodeToRGB8(up.image)
        try savePNG(rgb: out.pixels, width: out.width, height: out.height, to: outPath)
    } else {
        try up.image.data.write(to: URL(fileURLWithPath: outPath))
        out = try loadRGB8(outPath)
    }
    var lum = 0.0; for i in stride(from: 0, to: out.pixels.count, by: 3) { lum += Double(out.pixels[i]) * 0.299 + Double(out.pixels[i + 1]) * 0.587 + Double(out.pixels[i + 2]) * 0.114 }
    log(String(format: "engine: register %.2fs · run %.1fs (%@ %d×%d → %d×%d, appliedScale %d, %@) · %d tile reports (last %@) · mean luma %.1f · peak MLX %.2f GB",
               tReg, tRun, quant.rawValue as NSString, img.width, img.height, out.width, out.height, up.appliedScale,
               up.image.format.rawValue as NSString, tiles.count, (tiles.last ?? "-") as NSString, lum / Double(out.pixels.count / 3), Double(Memory.peakMemory) / 1e9))
    await engine.evict(package: id)
}

// MARK: - main

let args = Array(CommandLine.arguments.dropFirst())
do {
    guard let mode = args.first else { throw SmokeError.usage("vosr-smoke keys|gate|run …") }
    switch mode {
    case "keys":
        guard args.count >= 2 else { throw SmokeError.usage("vosr-smoke keys <weightsDir> [fp32|fp16]") }
        let lane = args.count > 2 ? args[2] : "fp32"
        try Device.withDefaultDevice(Device(.cpu)) { try runKeys(URL(fileURLWithPath: args[1]), lane: lane) }
    case "gate":
        guard args.count >= 3 else { throw SmokeError.usage("vosr-smoke gate <weightsDir> <goldens.safetensors> [--only a,b] [--gpu]") }
        var only = Set<String>(); var gpu = false
        var i = 3
        while i < args.count {
            if args[i] == "--only", i + 1 < args.count { only = Set(args[i + 1].split(separator: ",").map(String.init)); i += 2 }
            else if args[i] == "--gpu" { gpu = true; i += 1 } else { i += 1 }
        }
        let body = { try runGate(URL(fileURLWithPath: args[1]), goldens: URL(fileURLWithPath: args[2]), only: only, gpu: gpu) }
        if gpu { try body() } else { try Device.withDefaultDevice(Device(.cpu), body) }
    case "run":
        guard args.count >= 4 else { throw SmokeError.usage("vosr-smoke run <in.png> <out.png> <weightsDir> [--scale N] [--fp16|--bf16] [--int8] [--seed N] [--cpu] [--vae-tile N]") }
        var scale = 1, dtype = "fp32", int8 = false, seed: UInt64 = 0, cpu = false, vaeTile: Int? = nil, cacheLimit: Int? = nil
        var i = 4
        while i < args.count {
            switch args[i] {
            case "--scale": scale = Int(args[i + 1]) ?? 1; i += 2
            case "--bf16": dtype = "bf16"; i += 1
            case "--fp16": dtype = "fp16"; i += 1
            case "--int8": int8 = true; i += 1
            case "--seed": seed = UInt64(args[i + 1]) ?? 0; i += 2
            case "--cpu": cpu = true; i += 1
            case "--vae-tile": vaeTile = Int(args[i + 1]); i += 2
            case "--cache-limit": cacheLimit = Int(args[i + 1]); i += 2   // the engine's MLX pool cap, on the raw path (diagnostic)
            default: i += 1
            }
        }
        if let cacheLimit { GPU.set(cacheLimit: cacheLimit); log("MLX cache limit \(cacheLimit) B") }
        let body = { try runImage(args[1], args[2], URL(fileURLWithPath: args[3]), scale: scale, dtype: dtype, int8: int8, seed: seed, vaeTile: vaeTile) }
        if cpu { try Device.withDefaultDevice(Device(.cpu), body) } else { try body() }
    case "engine":
        guard args.count >= 4 else { throw SmokeError.usage("vosr-smoke engine <in.png> <out.png> <weightsDir> [--scale N] [--int8|--fp32] [--raw] [--vae-whole-frame PX]") }
        var scale: Int? = nil, quant = Quant.fp16, raw = false, wholeFrame: Int? = nil
        var i = 4
        while i < args.count {
            switch args[i] {
            case "--scale": scale = Int(args[i + 1]); i += 2
            case "--int8": quant = .int8; i += 1
            case "--fp32": quant = .fp32; i += 1
            case "--raw": raw = true; i += 1
            case "--vae-whole-frame": wholeFrame = Int(args[i + 1]); i += 2   // output pixels decoded whole-frame (AB-D-0120 lever)
            default: i += 1
            }
        }
        // Top-level `await` (SE-0343): the engine hops through main-actor monitors, so the main thread must be
        // free to service them — never block it behind a semaphore here (that deadlocks on the first hop).
        try await runEngine(args[1], args[2], URL(fileURLWithPath: args[3]), scale: scale, quant: quant, raw: raw, vaeWholeFramePixels: wholeFrame)
    default:
        throw SmokeError.usage("vosr-smoke keys|gate|run|engine …")
    }
} catch {
    log("\(error)")
    exit(1)
}
// The engine's actors and MLX's Metal threads can keep the process alive after top-level code returns
// (observed: a finished `engine` run lingering at 0 % CPU); leave explicitly.
exit(0)
