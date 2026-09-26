// DINOv2.swift — Swift/MLX port of the DINOv2 ViT-L/14 encoder VOSR conditions on (facebookresearch/dinov2,
// Apache-2.0), truncated at the block the DiT consumes (`layer_dinov2b_list = [17]`).
//
// forward_with_features (inference_vosr_onestep.py): prepare_tokens_with_masks (patch conv 14×14 stride 14 →
// prepend cls → + pos_embed) → blocks 0…17 → features['layer_17'] = x[:, 1:] (patch tokens, BEFORE the model's
// final norm). Block = affine LN (eps 1e-6) → qkv attention (bias) → LayerScale, then LN → GELU-exact MLP →
// LayerScale, both residual. No register tokens (the non-`reg` checkpoint), no RoPE (learned pos_embed).
//
// pos_embed: the checkpoint's 37×37 grid is bicubically interpolated to 32×32 (`interpolate_pos_encoding`,
// interpolate_offset 0.1) for the pipeline's fixed 448² input; the converter precomputes that table and ships
// it under the same key as [1, 1025, 1024], so this module asserts a 448² input.

import Foundation
import MLX
import MLXNN

/// `layers/attention.py` `Attention` (fused qkv with bias, proj with bias).
public final class DINOv2Attention: Module {
    let qkv: Linear
    let proj: Linear
    let numHeads: Int
    let headDim: Int

    public init(dim: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        qkv = Linear(dim, dim * 3, bias: true)
        proj = Linear(dim, dim, bias: true)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, N, C) = (x.dim(0), x.dim(1), x.dim(2))
        // reshape (B, N, 3, heads, hd) → unbind(2) → transpose(1, 2): [B, heads, N, hd]
        let qkv = qkv(x).reshaped([B, N, 3, numHeads, headDim])
        let q = qkv[0..., 0..., 0].transposed(0, 2, 1, 3)
        let k = qkv[0..., 0..., 1].transposed(0, 2, 1, 3)
        let v = qkv[0..., 0..., 2].transposed(0, 2, 1, 3)
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                  scale: 1.0 / Float(headDim).squareRoot(), mask: nil)
        return proj(o.transposed(0, 2, 1, 3).reshaped([B, N, C]))
    }
}

/// `layers/layer_scale.py` `LayerScale`: `x * gamma`.
public final class LayerScale: Module, UnaryLayer {
    let gamma: MLXArray
    public init(dim: Int) {
        gamma = MLXArray.ones([dim])
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { x * gamma }
}

/// timm / dinov2 `Mlp`: fc1 → act → fc2. `tanhGELU` selects `nn.GELU(approximate="tanh")` (the DiT's
/// `mlp_ca`); the default is the exact erf GELU (DINOv2).
public final class Mlp: Module, UnaryLayer {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    let tanhGELU: Bool

    public init(inFeatures: Int, hiddenFeatures: Int, outFeatures: Int? = nil, tanhGELU: Bool = false) {
        self._fc1.wrappedValue = Linear(inFeatures, hiddenFeatures, bias: true)
        self._fc2.wrappedValue = Linear(hiddenFeatures, outFeatures ?? inFeatures, bias: true)
        self.tanhGELU = tanhGELU
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = fc1(x)
        return fc2(tanhGELU ? geluApproximate(h) : gelu(h))
    }
}

/// `layers/block.py` `Block` (eval path): x + ls1(attn(norm1(x))); x + ls2(mlp(norm2(x))).
public final class DINOv2Block: Module, UnaryLayer {
    let norm1: LayerNorm
    let attn: DINOv2Attention
    let ls1: LayerScale
    let norm2: LayerNorm
    let mlp: Mlp
    let ls2: LayerScale

    public init(dim: Int, numHeads: Int, mlpRatio: Int = 4, eps: Float = 1e-6) {
        norm1 = LayerNorm(dimensions: dim, eps: eps, affine: true)
        attn = DINOv2Attention(dim: dim, numHeads: numHeads)
        ls1 = LayerScale(dim: dim)
        norm2 = LayerNorm(dimensions: dim, eps: eps, affine: true)
        mlp = Mlp(inFeatures: dim, hiddenFeatures: dim * mlpRatio)
        ls2 = LayerScale(dim: dim)
        super.init()
    }

    public func callAsFunction(_ xIn: MLXArray) -> MLXArray {
        var x = xIn + ls1(attn(norm1(xIn)))
        x = x + ls2(mlp(norm2(x)))
        return x
    }
}

/// `layers/patch_embed.py` `PatchEmbed` (no norm): conv(patch, stride patch) → flatten(2) → transpose → [B, N, D].
/// Takes channels-last pixels; token order is row-major (h * nW + w), exactly torch's flatten of (B, D, h, w).
public final class DINOv2PatchEmbed: Module {
    public let proj: Conv2d
    let patchSize: Int

    public init(patchSize: Int = 14, inChannels: Int = 3, embedDim: Int = 1024) {
        self.patchSize = patchSize
        proj = Conv2d(inputChannels: inChannels, outputChannels: embedDim, kernelSize: IntOrPair(patchSize),
                      stride: IntOrPair(patchSize), padding: 0)
        super.init()
    }

    public func callAsFunction(_ nhwc: MLXArray) -> MLXArray {
        let y = proj(nhwc)                                            // [B, h, w, D]
        return y.reshaped([y.dim(0), y.dim(1) * y.dim(2), y.dim(3)])
    }
}

/// DINOv2 ViT-L/14, blocks 0…`depth-1` only, feature output = patch tokens after block `featureLayer`.
public final class DINOv2ViT: Module {
    @ParameterInfo(key: "cls_token") var clsToken: MLXArray
    /// Pre-interpolated for the 448² input: [1, 1 + 32·32, 1024].
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray
    @ModuleInfo(key: "patch_embed") public var patchEmbed: DINOv2PatchEmbed
    public let blocks: [DINOv2Block]
    public let embedDim: Int
    public let patchSize: Int
    public let inputSize: Int
    public let featureLayer: Int

    public static let imagenetMean: [Float] = [0.485, 0.456, 0.406]
    public static let imagenetStd: [Float] = [0.229, 0.224, 0.225]

    public init(embedDim: Int = 1024, depth: Int = 18, numHeads: Int = 16, patchSize: Int = 14,
                inputSize: Int = 448, featureLayer: Int = 17) {
        precondition(featureLayer < depth, "featureLayer must be inside the kept block range")
        self.embedDim = embedDim
        self.patchSize = patchSize
        self.inputSize = inputSize
        self.featureLayer = featureLayer
        let grid = inputSize / patchSize
        self._clsToken.wrappedValue = MLXArray.zeros([1, 1, embedDim])
        self._posEmbed.wrappedValue = MLXArray.zeros([1, 1 + grid * grid, embedDim])
        self._patchEmbed.wrappedValue = DINOv2PatchEmbed(patchSize: patchSize, inChannels: 3, embedDim: embedDim)
        blocks = (0..<depth).map { _ in DINOv2Block(dim: embedDim, numHeads: numHeads) }
        super.init()
        train(false)   // C14
    }

    public var dtype: DType { patchEmbed.proj.weight.dtype }

    /// `preprocess_raw_image` on a [B, 3, H, W] image already in [0, 1] (the reference divides its
    /// 0…255 input by 255 first — the pipeline hands us `0.5 * lq + 0.5`): torch-bicubic to `inputSize`,
    /// clip to [0, 1], ImageNet normalise.
    public func preprocess(_ x01: MLXArray) -> MLXArray {
        var x = TorchBicubic.interpolate(x01.asType(.float32), toHeight: inputSize, toWidth: inputSize)
        x = clip(x, min: 0.0, max: 1.0)
        let mean = MLXArray(Self.imagenetMean, [1, 3, 1, 1])
        let std = MLXArray(Self.imagenetStd, [1, 3, 1, 1])
        return (x - mean) / std
    }

    /// `prepare_tokens_with_masks` (no masks, no register tokens): [B, 1 + N, D].
    public func prepareTokens(_ pixels: MLXArray) -> MLXArray {
        precondition(pixels.dim(2) == inputSize && pixels.dim(3) == inputSize,
                     "DINOv2ViT: pos_embed is precomputed for \(inputSize)² inputs, got \(pixels.dim(2))×\(pixels.dim(3))")
        let x = patchEmbed(pixels.asType(dtype).transposed(0, 2, 3, 1))
        let B = x.dim(0)
        let cls = broadcast(clsToken.asType(x.dtype), to: [B, 1, embedDim])
        return concatenated([cls, x], axis: 1) + posEmbed.asType(x.dtype)
    }

    /// Hidden states after block `upTo` (inclusive), cls token included: [B, 1 + N, D].
    public func hiddenStates(_ pixels: MLXArray, upTo layer: Int) -> MLXArray {
        var h = prepareTokens(pixels)
        for i in 0...layer { h = blocks[i](h) }
        return h
    }

    /// `features['layer_17']` = block-17 output without the cls token: [B, N, D]. Input = normalised pixels.
    public func features(_ pixels: MLXArray) -> MLXArray {
        hiddenStates(pixels, upTo: featureLayer)[0..., 1..., 0...]
    }
}
