// VOSR2Pipeline.swift — the VOSR one-step tiled driver (`inference_vosr_onestep.py::tiled_latent_inference`
// + `tiled_vae.py`) over the three ported components, and the exact-coverage weight loader.
//
// Pipeline (upscale u, image already bicubic-upsampled ×u by the caller — PIL semantics, `PILBicubic`):
//   lq ∈ [−1, 1] NCHW → VAE encode (full image, posterior mode, normalised) → latent tiles of 64 (= 512 px)
//   with overlap 8 → per tile: DINOv2 features from the LQ PIXEL crop resized to 448² → DiT velocity on
//   [lq_latent ‖ noise] at t = 1 → Gaussian-weighted blend → z = noise − u → VAE decode (full image).
// One shared noise draw per image (MLX normal; parity gates inject the torch noise), one Euler step.

import Foundation
import MLX
import MLXNN
import MLXRandom

public enum VOSRError: Error, CustomStringConvertible {
    case weightsNotFound(String)
    case loadFailed(String)
    /// The checkpoint does not cover the module tree exactly — a silently partial load is the failure
    /// mode that produces plausible-but-wrong output, so it is loud.
    case parameterMismatch(component: String, missing: [String], unused: [String])
    case badInput(String)

    public var description: String {
        switch self {
        case .weightsNotFound(let p): return "VOSR weights not found: \(p)"
        case .loadFailed(let d): return "VOSR weight load failed: \(d)"
        case .parameterMismatch(let c, let m, let u):
            return "VOSR \(c) key contract broken — missing \(m.count) \(m.prefix(5)), unused \(u.count) \(u.prefix(5))"
        case .badInput(let d): return "VOSR bad input: \(d)"
        }
    }
}

// MARK: - exact-coverage loading

extension Module {
    /// Load a flat `[key: array]` dictionary into this module with the key contract enforced:
    /// 0 missing, 0 unused, shapes verified. `cast` optionally converts every array's dtype first.
    public func loadExact(_ sd: [String: MLXArray], component: String, cast: DType? = nil) throws {
        let expected = Set(parameters().flattened().map { $0.0 })
        let provided = Set(sd.keys)
        if expected != provided {
            throw VOSRError.parameterMismatch(component: component,
                                              missing: expected.subtracting(provided).sorted(),
                                              unused: provided.subtracting(expected).sorted())
        }
        let flat: [(String, MLXArray)] = sd.map { (k, v) in (k, cast.map { v.asType($0) } ?? v) }
        do {
            try update(parameters: ModuleParameters.unflattened(flat), verify: .all)
        } catch {
            throw VOSRError.loadFailed("\(component): \(error)")
        }
    }

    /// `loadExact` from a safetensors file (read on the CPU stream, as the fleet doctrine requires).
    public func loadExact(url: URL, component: String, cast: DType? = nil) throws {
        guard FileManager.default.fileExists(atPath: url.path) else { throw VOSRError.weightsNotFound(url.path) }
        let sd: [String: MLXArray]
        do { sd = try MLX.loadArrays(url: url) } catch { throw VOSRError.loadFailed("\(component): \(error)") }
        try loadExact(sd, component: component, cast: cast)
    }

    /// Total parameter element count.
    public var parameterCount: Int {
        parameters().flattened().reduce(0) { $0 + $1.1.size }
    }
}

// MARK: - the driver

public final class VOSR2Pipeline {
    public let vae: AutoencoderKLQwenImage2D
    public let venc: DINOv2ViT
    public let model: LightningDiT

    /// Pixel tile / overlap for the DiT (`--tile_size 512 --tile_overlap 32`).
    public var tileSize = 512
    public var tileOverlap = 32
    /// VAE tile size in pixels; nil = full-image VAE (the reference default).
    public var vaeTileSize: Int? = nil
    /// Called after every DiT tile (tileIndex, tileCount) — progress reporting.
    public var onTile: ((Int, Int) -> Void)? = nil
    /// Called BEFORE every DiT tile and before the decode; a throw (e.g. `CancellationError`) is rethrown
    /// unchanged — the engine's cooperative-cancellation seam (CAN gate), one checkpoint per tile.
    public var checkpoint: (() throws -> Void)? = nil

    public static let aeFactor = 8

    public init(vae: AutoencoderKLQwenImage2D, venc: DINOv2ViT, model: LightningDiT) {
        self.vae = vae
        self.venc = venc
        self.model = model
    }

    // MARK: VAE dispatch (tiled_vae.py)

    /// `encode_dispatch`: normalised latent [B, 16, H/8, W/8] from `x` NCHW in [−1, 1] (H, W multiples of 8).
    public func encodeLatent(_ x: MLXArray) throws -> MLXArray {
        guard x.dim(2) % Self.aeFactor == 0, x.dim(3) % Self.aeFactor == 0 else {
            throw VOSRError.badInput("image dims must be multiples of 8 (got \(x.dim(2))×\(x.dim(3)))")
        }
        guard let ts = vaeTileSize, ts > 0 else { return vae.encodeNormalized(x) }
        let (lh, lw) = (x.dim(2) / Self.aeFactor, x.dim(3) / Self.aeFactor)
        let (ltSize, ltOverlap) = Self.tileParams(tileSize: ts, tileOverlap: tileOverlap, lh: lh, lw: lw)
        if lh <= ltSize && lw <= ltSize { return vae.encodeNormalized(x) }
        let hPos = makeTileGrid(length: lh, tile: ltSize, overlap: ltOverlap)
        let wPos = makeTileGrid(length: lw, tile: ltSize, overlap: ltOverlap)
        let g = gaussianWeights(tileH: ltSize, tileW: ltSize, channels: vae.zDim)
        let acc = MLXArray.zeros([x.dim(0), vae.zDim, lh, lw])
        let wacc = MLXArray.zeros([x.dim(0), vae.zDim, lh, lw])
        for hi in hPos {
            for wi in wPos {
                let crop = x[0..., 0..., (hi * 8)..<((hi + ltSize) * 8), (wi * 8)..<((wi + ltSize) * 8)]
                let zt = vae.encodeNormalized(crop).asType(.float32)
                acc[0..., 0..., hi..<(hi + ltSize), wi..<(wi + ltSize)] = acc[0..., 0..., hi..<(hi + ltSize), wi..<(wi + ltSize)] + zt * g
                wacc[0..., 0..., hi..<(hi + ltSize), wi..<(wi + ltSize)] = wacc[0..., 0..., hi..<(hi + ltSize), wi..<(wi + ltSize)] + g
                eval(acc, wacc)
            }
        }
        return acc / wacc
    }

    /// `decode_dispatch`: pixels NCHW in [−1, 1] from a normalised latent.
    public func decodeLatent(_ z: MLXArray) -> MLXArray { decodeLatent(z, with: vae, tileSize: vaeTileSize) }

    /// A decode other than `vae` at `vaeTileSize`. A 16-bit request decodes the final latent with the VAE's weights cast to
    /// float32 (AB-T-0233, AB-D-0127): the fp16 decode leaves a half-precision grid under a 16-bit output (AB-L-0238).
    /// Float32 activations double the decode's working set, so it carries its own, smaller tile size.
    public struct Decoder {
        public let vae: AutoencoderKLQwenImage2D
        public let tileSize: Int?
        public init(vae: AutoencoderKLQwenImage2D, tileSize: Int?) { self.vae = vae; self.tileSize = tileSize }
    }

    /// `decodeLatent` with another VAE and tile size (`nil`: the whole latent at once).
    public func decodeLatent(_ z: MLXArray, with vae: AutoencoderKLQwenImage2D, tileSize: Int?) -> MLXArray {
        guard let ts = tileSize, ts > 0 else { return vae.decodeNormalized(z) }
        let (lh, lw) = (z.dim(2), z.dim(3))
        let (ltSize, ltOverlap) = Self.tileParams(tileSize: ts, tileOverlap: ts / 8, lh: lh, lw: lw)
        if lh <= ltSize && lw <= ltSize { return vae.decodeNormalized(z) }
        let hPos = makeTileGrid(length: lh, tile: ltSize, overlap: ltOverlap)
        let wPos = makeTileGrid(length: lw, tile: ltSize, overlap: ltOverlap)
        let g = gaussianWeights(tileH: ltSize * 8, tileW: ltSize * 8, channels: 3)
        let acc = MLXArray.zeros([z.dim(0), 3, lh * 8, lw * 8])
        let wacc = MLXArray.zeros([z.dim(0), 3, lh * 8, lw * 8])
        for hi in hPos {
            for wi in wPos {
                let (he, we) = (hi + ltSize, wi + ltSize)
                let pix = vae.decodeNormalized(z[0..., 0..., hi..<he, wi..<we]).asType(.float32)
                acc[0..., 0..., (hi * 8)..<(he * 8), (wi * 8)..<(we * 8)] = acc[0..., 0..., (hi * 8)..<(he * 8), (wi * 8)..<(we * 8)] + pix * g
                wacc[0..., 0..., (hi * 8)..<(he * 8), (wi * 8)..<(we * 8)] = wacc[0..., 0..., (hi * 8)..<(he * 8), (wi * 8)..<(we * 8)] + g
                eval(acc, wacc)
            }
        }
        return clip(acc / wacc, min: -1.0, max: 1.0)
    }

    /// `_tile_params` (tiled_vae.py): latent tile size/overlap from a pixel tile size, capped to the latent.
    static func tileParams(tileSize: Int, tileOverlap: Int, lh: Int, lw: Int) -> (Int, Int) {
        var ltSize = max(tileSize / aeFactor, 1)
        var ltOverlap = max(tileOverlap / aeFactor, ltSize / 8)
        ltSize = min(ltSize, min(lh, lw))
        ltOverlap = min(ltOverlap, ltSize - 1)
        return (ltSize, ltOverlap)
    }

    // MARK: DINOv2 conditioning

    /// `get_venc_features` for an LQ pixel tensor in [−1, 1]: `(0.5·lq + 0.5)` → preprocess → layer-17 tokens.
    public func vencFeatures(_ lq: MLXArray) -> MLXArray {
        venc.features(venc.preprocess(0.5 * lq.asType(.float32) + 0.5))
    }

    // MARK: the one-step tiled flow

    /// Intermediates of one `superResolve` (for the parity gates).
    public struct Trace {
        public var latent: MLXArray
        public var noise: MLXArray
        public var velocity: MLXArray
        public var latentOut: MLXArray
    }

    /// `tiled_latent_inference` with `infer_steps = 1`: NCHW [−1, 1] in → NCHW [−1, 1] out, same size.
    /// `noise` (normalised-latent shaped, NCHW) overrides the seeded draw. `decoder` decodes the final latent (float32,
    /// for a 16-bit output) in place of `vae` at `vaeTileSize`; the encode and the DiT are the same either way.
    public func superResolve(_ lq: MLXArray, noise: MLXArray? = nil, seed: UInt64 = 0, decoder: Decoder? = nil) throws -> MLXArray {
        try superResolveTraced(lq, noise: noise, seed: seed, decoder: decoder).output
    }

    /// `superResolve` that also returns the intermediates (parity gates).
    public func superResolveTraced(_ lq: MLXArray, noise: MLXArray? = nil, seed: UInt64 = 0,
                                   decoder: Decoder? = nil) throws -> (output: MLXArray, trace: Trace) {
        let b = lq.dim(0)
        let lqLatent = try encodeLatent(lq)
        eval(lqLatent)
        let (lc, lh, lw) = (lqLatent.dim(1), lqLatent.dim(2), lqLatent.dim(3))
        let p = model.patchSize

        // Pixel tile params → latent tile params (aligned to patch size)
        var ltSize = max((tileSize / Self.aeFactor / p) * p, p)
        var ltOverlap = max(tileOverlap / Self.aeFactor, ltSize / 8)
        ltSize = min(ltSize, min(lh, lw))
        ltOverlap = min(ltOverlap, ltSize - 1)

        let z0 = noise ?? MLXRandom.normal(lqLatent.shape, key: MLXRandom.key(seed))
        let tCur = MLXArray([Float](repeating: 1.0, count: b))
        var velocity: MLXArray

        if lh <= ltSize && lw <= ltSize {
            // Fast path: no tiling needed (the whole latent is one DiT call).
            try checkpoint?()
            let zFea = vencFeatures(lq)
            velocity = model(concatenated([lqLatent, z0.asType(lqLatent.dtype)], axis: 1), t: tCur, z: zFea).asType(.float32)
            eval(velocity)
            onTile?(1, 1)
        } else {
            let hPos = makeTileGrid(length: lh, tile: ltSize, overlap: ltOverlap)
            let wPos = makeTileGrid(length: lw, tile: ltSize, overlap: ltOverlap)
            let g = gaussianWeights(tileH: ltSize, tileW: ltSize, channels: lc)
            let uAcc = MLXArray.zeros(lqLatent.shape)
            let wAcc = MLXArray.zeros(lqLatent.shape)
            let total = hPos.count * wPos.count
            var index = 0
            for hi in hPos {
                for wi in wPos {
                    try checkpoint?()
                    let (he, we) = (hi + ltSize, wi + ltSize)
                    // Per-tile DINOv2 features from the LQ pixel crop (constant 448² shape, as in training).
                    let phS = hi * Self.aeFactor, pwS = wi * Self.aeFactor
                    let phE = min(he * Self.aeFactor, lq.dim(2)), pwE = min(we * Self.aeFactor, lq.dim(3))
                    let zFeaTile = vencFeatures(lq[0..., 0..., phS..<phE, pwS..<pwE])
                    let inp = concatenated([lqLatent[0..., 0..., hi..<he, wi..<we],
                                            z0[0..., 0..., hi..<he, wi..<we].asType(lqLatent.dtype)], axis: 1)
                    let uTile = model(inp, t: tCur, z: zFeaTile).asType(.float32)
                    uAcc[0..., 0..., hi..<he, wi..<we] = uAcc[0..., 0..., hi..<he, wi..<we] + uTile * g
                    wAcc[0..., 0..., hi..<he, wi..<we] = wAcc[0..., 0..., hi..<he, wi..<we] + g
                    eval(uAcc, wAcc)
                    index += 1
                    onTile?(index, total)
                }
            }
            velocity = uAcc / wAcc
        }
        // z = z − (t_cur − t_nxt) · u with t: 1 → 0
        let z1 = z0.asType(.float32) - velocity
        try checkpoint?()
        // A deep decoder takes the float32 latent as it is; the default decode casts it to the lane's dtype, as before.
        let out = decoder.map { decodeLatent(z1, with: $0.vae, tileSize: $0.tileSize) } ?? decodeLatent(z1.asType(lqLatent.dtype))
        eval(out)
        return (out, Trace(latent: lqLatent, noise: z0, velocity: velocity, latentOut: z1))
    }
}

// MARK: - loading a full pipeline from the converter's output directory

public extension VOSR2Pipeline {
    /// Which converted files to load (`oracle/convert_weights.py` output layout).
    struct WeightFiles: Sendable {
        public var transformer = "transformer_fp32.safetensors"
        public var vae = "vae_fp32.safetensors"
        public var dinov2 = "dinov2_fp32.safetensors"
        public init() {}
        public static let fp32 = WeightFiles()
        /// The half-precision tier that matches the reference's own fp16 run (VOSR-EVAL fixtures: fp16 vs fp32
        /// 56–59 dB PSNR); bf16's 8-bit mantissa on a residual stream that reaches |126| measured ~38 dB.
        public static var fp16: WeightFiles {
            var f = WeightFiles(); f.transformer = "transformer_fp16.safetensors"; f.vae = "vae_fp16.safetensors"; f.dinov2 = "dinov2_fp16.safetensors"; return f
        }
        public static var bf16: WeightFiles {
            var f = WeightFiles(); f.transformer = "transformer_bf16.safetensors"; f.vae = "vae_fp16.safetensors"; f.dinov2 = "dinov2_fp16.safetensors"; return f
        }
    }

    /// Construct the three components and load them with the key contract enforced. `quantizeDiT` replaces
    /// every DiT `Linear` with an int8 `QuantizedLinear` (group size 64) after loading — the S6 tier. A
    /// quantized forward is Metal-only: run it on the GPU stream (a CPU pin grinds for hours, fleet lesson).
    static func load(from dir: URL, files: WeightFiles = .fp32, quantizeDiT: Bool = false) throws -> VOSR2Pipeline {
        let vae = AutoencoderKLQwenImage2D()
        try vae.loadExact(url: dir.appendingPathComponent(files.vae), component: "vae")
        let venc = DINOv2ViT()
        try venc.loadExact(url: dir.appendingPathComponent(files.dinov2), component: "dinov2")
        let model = LightningDiT()
        try model.loadExact(url: dir.appendingPathComponent(files.transformer), component: "transformer")
        if quantizeDiT { model.quantizeLinears(groupSize: 64, bits: 8) }
        return VOSR2Pipeline(vae: vae, venc: venc, model: model)
    }
}
