// QwenImageVAE2D.swift — Swift/MLX port of VOSR's `models/qwenimage_vae2d.py` (Apache-2.0; the Qwen-Image /
// Wan 2-D image VAE, diffusers naming). Isomorphic: same class names, same decomposition, same forward order.
//
// Layout: channels-last (B, H, W, C) internally — MLX-native for Conv2d — with one transpose at the
// encode()/decode() boundary so the pipeline-facing convention stays PyTorch's (B, C, H, W).
//
// RMSNorm2D is `F.normalize(x, dim=1) * sqrt(C) * gamma`: L2 over channels clamped at 1e-12, NOT
// mean-square RMSNorm. gamma is checkpointed (C,1,1) and squeezed to (C) by the converter.
//
// Every 3×3 conv is a `WinogradFreeConv2d` (AB-L-0163): inside mlx's Winograd window the GPU conv2d
// is lossy; the conv3d route is exact. Config of the shipped checkpoint: base 96, mult [1,2,4,4],
// 2 res blocks, z 16, no attention scales (mid-block attention only), silu.

import Foundation
import MLX
import MLXNN

// MARK: - layers

/// `RMSNorm2D`: `F.normalize(x, dim=channels) * dim**0.5 * gamma`. x is channels-last here.
public final class RMSNorm2D: Module, UnaryLayer {
    @ParameterInfo(key: "gamma") var gamma: MLXArray
    let scale: Float

    public init(dim: Int) {
        self._gamma.wrappedValue = MLXArray.ones([dim])
        self.scale = Float(dim).squareRoot()
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // The channel L2 is accumulated in fp32: mid-block activations reach |x| ≈ 160, so Σx² over 384
        // channels (~10⁷) overflows fp16, and 1e-12 is below the half-precision floor anyway.
        let xf = x.asType(.float32)
        let l2 = sqrt(sum(xf * xf, axis: -1, keepDims: true))
        let normalized = xf / maximum(l2, MLXArray(Float(1e-12))) * scale
        return normalized.asType(x.dtype) * gamma
    }
}

/// Parameter-free nearest 2× spatial upsample (index 0 of the upsample `resample` Sequential).
final class NearestUpsample2x: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2)
    }
}

/// `nn.ZeroPad2d((0, 1, 0, 1))` — one zero column on the right, one zero row at the bottom.
final class ZeroPad2dRightBottom: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        padded(x, widths: [IntOrPair([0, 0]), IntOrPair([0, 1]), IntOrPair([0, 1]), IntOrPair([0, 0])])
    }
}

func conv3x3(_ inDim: Int, _ outDim: Int, stride: Int = 1, padding: Int = 1) -> Conv2d {
    WinogradFreeConv2d(inputChannels: inDim, outputChannels: outDim, kernelSize: 3,
                       stride: IntOrPair(stride), padding: IntOrPair(padding))
}

/// `ResidualBlock2D`: norm → silu → conv → norm → silu → conv, plus a 1×1 shortcut when dims change.
public final class ResidualBlock2D: Module, UnaryLayer {
    let norm1: RMSNorm2D
    let conv1: Conv2d
    let norm2: RMSNorm2D
    let conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

    public init(inDim: Int, outDim: Int) {
        norm1 = RMSNorm2D(dim: inDim)
        conv1 = conv3x3(inDim, outDim)
        norm2 = RMSNorm2D(dim: outDim)
        conv2 = conv3x3(outDim, outDim)
        self._convShortcut.wrappedValue = inDim != outDim
            ? Conv2d(inputChannels: inDim, outputChannels: outDim, kernelSize: 1) : nil
        super.init()
    }

    public func callAsFunction(_ xIn: MLXArray) -> MLXArray {
        let h = convShortcut.map { $0(xIn) } ?? xIn
        var x = norm1(xIn)
        x = silu(x)
        x = conv1(x)
        x = norm2(x)
        x = silu(x)
        x = conv2(x)
        return x + h
    }
}

/// `AttentionBlock2D`: single-head self-attention over all spatial positions (mid block only).
public final class AttentionBlock2D: Module, UnaryLayer {
    let norm: RMSNorm2D
    @ModuleInfo(key: "to_qkv") var toQKV: Conv2d
    let proj: Conv2d
    let dim: Int

    public init(dim: Int) {
        self.dim = dim
        norm = RMSNorm2D(dim: dim)
        self._toQKV.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim * 3, kernelSize: 1)
        proj = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 1)
        super.init()
    }

    public func callAsFunction(_ xIn: MLXArray) -> MLXArray {
        let identity = xIn
        let (b, h, w, c) = (xIn.dim(0), xIn.dim(1), xIn.dim(2), xIn.dim(3))
        var x = norm(xIn)
        // upstream: qkv (b, 3c, h, w) → reshape (b, 3, c, hw) → permute (b, 3, hw, c). Channels-last:
        // (b, h, w, 3c) → (b, hw, 3, c) → (b, 3, hw, c); the 3c axis splits as (3, c) in both.
        let qkv = toQKV(x).reshaped([b, h * w, 3, c]).transposed(0, 2, 1, 3)
        let q = qkv[0..., 0].expandedDimensions(axis: 1)               // (b, 1, hw, c) — the 4-D single-head form
        let k = qkv[0..., 1].expandedDimensions(axis: 1)
        let v = qkv[0..., 2].expandedDimensions(axis: 1)
        let attn = Self.attention(q, k, v, scale: 1.0 / Float(c).squareRoot())
        x = proj(attn.squeezed(axis: 1).reshaped([b, h, w, c]))
        return x + identity
    }

    /// Queries above which the single-head attention runs query-chunked. The head dim here is the full channel width
    /// (384), outside mlx's fused-SDPA window, so the score matrix is materialised: N² · 2 B = 8.6 GB at a 256² latent
    /// (2048² output) and 137 GB at 512². Chunking the queries bounds it to `chunk · N · B` with bit-for-bit the same math
    /// (softmax is per query row). Below the threshold the plain SDPA call is kept, so the gated 512²/1024² paths are
    /// byte-identical to before.
    ///
    /// `VOSR_VAE_CHUNK=<queries>` overrides both the threshold and the chunk size (diagnostic lever — timing and memory
    /// attribution, 2026-10-10; `VOSR_VAE_CHUNK=70000` runs a 2048² decode unchunked).
    static let chunkOverride: Int? = getenv("VOSR_VAE_CHUNK").flatMap { Int(String(cString: $0)) }
    static let chunkedQueryThreshold: Int = chunkOverride ?? 8192

    /// The score bytes one chunk may materialise: 256 MiB, the tiled decode's known-good working set (two 8192-query
    /// chunks per 1024² tile at fp16). The loop evaluates every chunk, and under the engine's 2 GiB MLX cache cap a chunk
    /// near the cap's size comes out WRONG (8192 queries at N = 65 536: 1.07 GB of fp16 scores per chunk, 8.71 dB on a
    /// whole-frame 2048² decode; AB-L-0226). Sized by bytes, a chunk stays at an eighth of the cap at any N and dtype.
    static let chunkScoreBytes = 256 << 20

    /// Queries per chunk for `keys` keys of `elementBytes` each: as many as fit `chunkScoreBytes` of scores (8192 for the
    /// tiled path's N = 16 384 at fp16, so that path is unchanged; 2048 for a whole-frame 2048² decode).
    static func queriesPerChunk(keys: Int, elementBytes: Int) -> Int {
        if let chunkOverride { return chunkOverride }
        return max(1, chunkScoreBytes / max(1, keys * elementBytes))
    }

    static func attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, scale: Float) -> MLXArray {
        let n = q.dim(2)
        guard n > chunkedQueryThreshold else {
            return MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        }
        let kT = k.transposed(0, 1, 3, 2)
        let chunk = queriesPerChunk(keys: k.dim(2), elementBytes: q.dtype.size)
        var parts: [MLXArray] = []
        var start = 0
        while start < n {
            let end = min(start + chunk, n)
            let scores = matmul(q[0..., 0..., start..<end, 0...], kT) * scale
            let out = matmul(softmax(scores, axis: -1), v)
            eval(out)                                                  // release this chunk's score matrix
            parts.append(out)
            start = end
        }
        return concatenated(parts, axis: 2)
    }
}

/// `Resample2D`: `upsample2d` = nearest 2× + Conv(dim → dim/2); `downsample2d` = ZeroPad(0,1,0,1) + stride-2 conv.
/// Checkpoint keys live at `resample.1` (index 1 of the Sequential) → `resample.layers.1` here.
public final class Resample2D: Module, UnaryLayer {
    let resample: Sequential

    public init(dim: Int, mode: String) {
        switch mode {
        case "upsample2d":
            resample = Sequential(layers: [NearestUpsample2x(), conv3x3(dim, dim / 2)])
        case "downsample2d":
            resample = Sequential(layers: [ZeroPad2dRightBottom(), conv3x3(dim, dim, stride: 2, padding: 0)])
        default:
            fatalError("Resample2D: unsupported mode \(mode)")
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { resample(x) }
}

/// `MidBlock2D`: resnet → (attention → resnet) × num_layers.
public final class MidBlock2D: Module, UnaryLayer {
    let attentions: [AttentionBlock2D]
    let resnets: [ResidualBlock2D]

    public init(dim: Int, numLayers: Int = 1) {
        var res = [ResidualBlock2D(inDim: dim, outDim: dim)]
        var att: [AttentionBlock2D] = []
        for _ in 0..<numLayers {
            att.append(AttentionBlock2D(dim: dim))
            res.append(ResidualBlock2D(inDim: dim, outDim: dim))
        }
        attentions = att
        resnets = res
        super.init()
    }

    public func callAsFunction(_ xIn: MLXArray) -> MLXArray {
        var x = resnets[0](xIn)
        for (attn, resnet) in zip(attentions, resnets.dropFirst()) {
            x = attn(x)
            x = resnet(x)
        }
        return x
    }
}

/// `Encoder2D`: conv_in → flat down_blocks (res, res, [down]) × levels → mid → norm → silu → conv_out.
public final class Encoder2D: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "down_blocks") var downBlocks: [UnaryLayer]
    @ModuleInfo(key: "mid_block") var midBlock: MidBlock2D
    @ModuleInfo(key: "norm_out") var normOut: RMSNorm2D
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    public init(dim: Int = 96, zDim: Int = 32, dimMult: [Int] = [1, 2, 4, 4], numResBlocks: Int = 2) {
        let dims = ([1] + dimMult).map { dim * $0 }
        self._convIn.wrappedValue = conv3x3(3, dims[0])
        var blocks: [UnaryLayer] = []
        for i in 0..<(dims.count - 1) {
            var inDim = dims[i]
            let outDim = dims[i + 1]
            for _ in 0..<numResBlocks {
                blocks.append(ResidualBlock2D(inDim: inDim, outDim: outDim))
                inDim = outDim
            }
            if i != dimMult.count - 1 { blocks.append(Resample2D(dim: outDim, mode: "downsample2d")) }
        }
        self._downBlocks.wrappedValue = blocks
        let last = dims[dims.count - 1]
        self._midBlock.wrappedValue = MidBlock2D(dim: last, numLayers: 1)
        self._normOut.wrappedValue = RMSNorm2D(dim: last)
        self._convOut.wrappedValue = conv3x3(last, zDim)
        super.init()
    }

    public func callAsFunction(_ xIn: MLXArray) -> MLXArray {
        var x = convIn(xIn)
        for layer in downBlocks { x = layer(x) }
        x = midBlock(x)
        x = normOut(x)
        x = silu(x)
        x = convOut(x)
        return x
    }
}

/// `UpBlock2D`: (num_res_blocks + 1) resnets, then an optional upsampler.
public final class UpBlock2D: Module, UnaryLayer {
    let resnets: [ResidualBlock2D]
    @ModuleInfo(key: "upsamplers") var upsamplers: [Resample2D]?

    public init(inDim: Int, outDim: Int, numResBlocks: Int, upsampleMode: String?) {
        var res: [ResidualBlock2D] = []
        var cur = inDim
        for _ in 0..<(numResBlocks + 1) {
            res.append(ResidualBlock2D(inDim: cur, outDim: outDim))
            cur = outDim
        }
        resnets = res
        self._upsamplers.wrappedValue = upsampleMode.map { [Resample2D(dim: outDim, mode: $0)] }
        super.init()
    }

    public func callAsFunction(_ xIn: MLXArray) -> MLXArray {
        var x = xIn
        for resnet in resnets { x = resnet(x) }
        if let upsamplers { x = upsamplers[0](x) }
        return x
    }
}

/// `Decoder2D`: conv_in → mid → up_blocks → norm → silu → conv_out.
public final class Decoder2D: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "mid_block") var midBlock: MidBlock2D
    @ModuleInfo(key: "up_blocks") var upBlocks: [UpBlock2D]
    @ModuleInfo(key: "norm_out") var normOut: RMSNorm2D
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    public init(dim: Int = 96, zDim: Int = 16, dimMult: [Int] = [1, 2, 4, 4], numResBlocks: Int = 2) {
        let dims = ([dimMult[dimMult.count - 1]] + dimMult.reversed()).map { dim * $0 }
        self._convIn.wrappedValue = conv3x3(zDim, dims[0])
        self._midBlock.wrappedValue = MidBlock2D(dim: dims[0], numLayers: 1)
        var ups: [UpBlock2D] = []
        for i in 0..<(dims.count - 1) {
            var inDim = dims[i]
            let outDim = dims[i + 1]
            if i > 0 { inDim = inDim / 2 }
            let mode: String? = i != dimMult.count - 1 ? "upsample2d" : nil
            ups.append(UpBlock2D(inDim: inDim, outDim: outDim, numResBlocks: numResBlocks, upsampleMode: mode))
        }
        self._upBlocks.wrappedValue = ups
        let last = dims[dims.count - 1]
        self._normOut.wrappedValue = RMSNorm2D(dim: last)
        self._convOut.wrappedValue = conv3x3(last, 3)
        super.init()
    }

    public func callAsFunction(_ xIn: MLXArray) -> MLXArray {
        var x = convIn(xIn)
        x = midBlock(x)
        for up in upBlocks { x = up(x) }
        x = normOut(x)
        x = silu(x)
        x = convOut(x)
        return x
    }
}

// MARK: - the autoencoder

/// `AutoencoderKLQwenImage2D` — encode to the deterministic posterior mode, decode with clamp(−1, 1).
/// Pipeline-facing tensors are NCHW in [−1, 1]; latents are NCHW [B, 16, H/8, W/8].
public final class AutoencoderKLQwenImage2D: Module {
    public let encoder: Encoder2D
    @ModuleInfo(key: "quant_conv") var quantConv: Conv2d
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d
    public let decoder: Decoder2D
    public let zDim: Int
    /// `latents_mean` / `latents_std` from the shipped config (Qwen-Image). Kept as plain arrays so the Module
    /// reflection does not count them as parameters (they are config, not checkpoint tensors).
    public let latentsMeanValues: [Float]
    public let latentsStdValues: [Float]
    /// [1, 16, 1, 1] broadcast forms.
    public var latentsMean: MLXArray { MLXArray(latentsMeanValues, [1, zDim, 1, 1]) }
    public var latentsStd: MLXArray { MLXArray(latentsStdValues, [1, zDim, 1, 1]) }

    public static let defaultLatentsMean: [Float] = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                                                     0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
    public static let defaultLatentsStd: [Float] = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                                                    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]

    public init(baseDim: Int = 96, zDim: Int = 16, dimMult: [Int] = [1, 2, 4, 4], numResBlocks: Int = 2,
                latentsMean: [Float] = AutoencoderKLQwenImage2D.defaultLatentsMean,
                latentsStd: [Float] = AutoencoderKLQwenImage2D.defaultLatentsStd) {
        self.zDim = zDim
        encoder = Encoder2D(dim: baseDim, zDim: zDim * 2, dimMult: dimMult, numResBlocks: numResBlocks)
        self._quantConv.wrappedValue = Conv2d(inputChannels: zDim * 2, outputChannels: zDim * 2, kernelSize: 1)
        self._postQuantConv.wrappedValue = Conv2d(inputChannels: zDim, outputChannels: zDim, kernelSize: 1)
        decoder = Decoder2D(dim: baseDim, zDim: zDim, dimMult: dimMult, numResBlocks: numResBlocks)
        self.latentsMeanValues = latentsMean
        self.latentsStdValues = latentsStd
        super.init()
        train(false)   // C14: born in inference mode (Dropout(0) upstream is never applied)
    }

    /// The parameter dtype (inputs are cast to it at the boundary).
    public var dtype: DType { quantConv.weight.dtype }

    /// `encoder(x)` → `quant_conv` → [B, 32, h, w] moments (NCHW). Exposed for the parity gates.
    public func moments(_ x: MLXArray) -> MLXArray {
        var h = encoder(x.asType(dtype).transposed(0, 2, 3, 1))
        h = quantConv(h)
        return h.transposed(0, 3, 1, 2)
    }

    /// `encode(x).latent_dist.mode()` — the mean half of the moments. Raw (un-normalised) latent, NCHW.
    public func encodeMode(_ x: MLXArray) -> MLXArray {
        moments(x)[0..., 0..<zDim]
    }

    /// `tiled_vae.encode_latent` (qwen, posterior_mode): `(mode − latents_mean) / latents_std`.
    public func encodeNormalized(_ x: MLXArray) -> MLXArray {
        let z = encodeMode(x)
        return (z - latentsMean.asType(z.dtype)) * (1.0 / latentsStd.asType(z.dtype))
    }

    /// `post_quant_conv` → `decoder` → clamp(−1, 1) on an UN-normalised latent (NCHW in, NCHW out).
    public func decode(_ z: MLXArray) -> MLXArray {
        var x = postQuantConv(z.asType(dtype).transposed(0, 2, 3, 1))
        x = decoder(x)
        return clip(x.transposed(0, 3, 1, 2), min: -1.0, max: 1.0)
    }

    /// `tiled_vae.decode_latent` (qwen): `latent / (1/std) + mean` then `decode`, clamped.
    public func decodeNormalized(_ z: MLXArray) -> MLXArray {
        let raw = z / (1.0 / latentsStd.asType(z.dtype)) + latentsMean.asType(z.dtype)
        return decode(raw)
    }
}
