// LightningDiT.swift — Swift/MLX port of VOSR's `models/lightningdit.py` (+ `swiglu_ffn.py`, `rmsnorm.py`,
// `pos_embed.py`; Apache-2.0 / MIT), the VOSR2 configuration: dim 1536, depth 36, 24 heads (head 64),
// patch 2, SwiGLU (hidden 4096), RMSNorm (eps 1e-6, fp32), QK-norm per head, EVA 2-D RoPE, adaLN-single
// (`scale_shift_table` + shared `t_block`), per-block cross-attention to DINOv2 tokens projected by a
// shared `layer_norm` + `mlp_ca`. `auxiliary_time_cond = false` (no `r_embedder`).
//
// Isomorphic to upstream (class names, decomposition, forward order). Forward = `forward_flexible`:
// square inputs, the checkpoint's 32×32 RoPE buffers when the patch grid is 32, dynamic tables otherwise.
// Input is NCHW [B, 32, H, W] (LQ latent ‖ noise); output NCHW [B, 16, H, W] (the flow velocity).

import Foundation
import MLX
import MLXNN

/// timm `PatchEmbed` (strict_img_size=False, no norm): conv(p, stride p) → flatten → [B, N, D].
public final class PatchEmbed: Module {
    public let proj: Conv2d
    public let patchSize: Int

    public init(patchSize: Int, inChannels: Int, embedDim: Int) {
        self.patchSize = patchSize
        proj = Conv2d(inputChannels: inChannels, outputChannels: embedDim, kernelSize: IntOrPair(patchSize),
                      stride: IntOrPair(patchSize), padding: 0)
        super.init()
    }

    /// nhwc → [B, h·w, D] with token index h * w + w (torch flatten(2).transpose(1, 2)).
    public func callAsFunction(_ nhwc: MLXArray) -> MLXArray {
        let y = proj(nhwc)
        return y.reshaped([y.dim(0), y.dim(1) * y.dim(2), y.dim(3)])
    }
}

/// `TimestepEmbedder`: sinusoidal(256) → Linear → SiLU → Linear. Keys `mlp.layers.{0,2}`.
public final class TimestepEmbedder: Module {
    let mlp: Sequential
    public let frequencyEmbeddingSize: Int

    public init(hiddenSize: Int, frequencyEmbeddingSize: Int = 256) {
        self.frequencyEmbeddingSize = frequencyEmbeddingSize
        mlp = Sequential(layers: [Linear(frequencyEmbeddingSize, hiddenSize, bias: true), SiLU(),
                                  Linear(hiddenSize, hiddenSize, bias: true)])
        super.init()
    }

    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        let tFreq = timestepEmbedding(t, dim: frequencyEmbeddingSize)
        return mlp(tFreq.asType(mlpDtype))
    }

    var mlpDtype: DType { (mlp.layers[0] as! Linear).weight.dtype }
}

/// `VisionRotaryEmbeddingFast` — holds the checkpoint's cos/sin buffers for the training grid (32×32,
/// [1024, 64]); `tables(for:)` returns them or recomputes for another square grid (`_get_dynamic_rope`).
public final class VisionRotaryEmbeddingFast: Module {
    @ParameterInfo(key: "freqs_cos") var freqsCos: MLXArray
    @ParameterInfo(key: "freqs_sin") var freqsSin: MLXArray
    public let dim: Int          // half the head dim (32)
    public let ptSeqLen: Int     // training grid side (32)

    public init(dim: Int, ptSeqLen: Int) {
        self.dim = dim
        self.ptSeqLen = ptSeqLen
        let (c, s) = visionRotaryTables(hwSeqLen: ptSeqLen, ptSeqLen: ptSeqLen, dim: dim)
        self._freqsCos.wrappedValue = c
        self._freqsSin.wrappedValue = s
        super.init()
    }

    /// (cos, sin) for an `hw × hw` patch grid in the given dtype.
    public func tables(for hw: Int, dtype: DType) -> (MLXArray, MLXArray) {
        if hw == ptSeqLen { return (freqsCos.asType(dtype), freqsSin.asType(dtype)) }
        let (c, s) = visionRotaryTables(hwSeqLen: hw, ptSeqLen: ptSeqLen, dim: dim)
        return (c.asType(dtype), s.asType(dtype))
    }
}

/// `Attention` (self): fused qkv with bias, per-head RMSNorm on q and k, RoPE, SDPA, proj.
public final class Attention: Module {
    @ModuleInfo var qkv: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo var proj: Linear
    let numHeads: Int
    let headDim: Int

    public init(dim: Int, numHeads: Int, eps: Float = 1e-6) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self._qkv.wrappedValue = Linear(dim, dim * 3, bias: true)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._proj.wrappedValue = Linear(dim, dim, bias: true)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, rope: (cos: MLXArray, sin: MLXArray)?) -> MLXArray {
        let (B, N, C) = (x.dim(0), x.dim(1), x.dim(2))
        // reshape (B, N, 3, heads, hd) → permute (2, 0, 3, 1, 4) → unbind(0)
        let qkv = qkv(x).reshaped([B, N, 3, numHeads, headDim]).transposed(2, 0, 3, 1, 4)
        var q = qNorm(qkv[0])
        var k = kNorm(qkv[1])
        let v = qkv[2]
        if let rope {
            q = applyVisionRope(q, cos: rope.cos, sin: rope.sin)
            k = applyVisionRope(k, cos: rope.cos, sin: rope.sin)
        }
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                  scale: 1.0 / Float(headDim).squareRoot(), mask: nil)
        return proj(o.transposed(0, 2, 1, 3).reshaped([B, N, C]))
    }
}

/// `MultiHeadCrossAttention`: q from x, k/v from cond, per-head RMSNorm on q and k, no RoPE.
public final class MultiHeadCrossAttention: Module {
    @ModuleInfo(key: "q_linear") var qLinear: Linear
    @ModuleInfo(key: "k_linear") var kLinear: Linear
    @ModuleInfo(key: "v_linear") var vLinear: Linear
    @ModuleInfo var proj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    let numHeads: Int
    let headDim: Int

    public init(dModel: Int, numHeads: Int, eps: Float = 1e-6) {
        self.numHeads = numHeads
        self.headDim = dModel / numHeads
        self._qLinear.wrappedValue = Linear(dModel, dModel, bias: true)
        self._kLinear.wrappedValue = Linear(dModel, dModel, bias: true)
        self._vLinear.wrappedValue = Linear(dModel, dModel, bias: true)
        self._proj.wrappedValue = Linear(dModel, dModel, bias: true)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, cond: MLXArray) -> MLXArray {
        let (B, N, C) = (x.dim(0), x.dim(1), x.dim(2))
        let Nc = cond.dim(1)
        var q = qLinear(x).reshaped([B, N, numHeads, headDim]).transposed(0, 2, 1, 3)
        var k = kLinear(cond).reshaped([B, Nc, numHeads, headDim]).transposed(0, 2, 1, 3)
        let v = vLinear(cond).reshaped([B, Nc, numHeads, headDim]).transposed(0, 2, 1, 3)
        q = qNorm(q)
        k = kNorm(k)
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                  scale: 1.0 / Float(headDim).squareRoot(), mask: nil)
        return proj(o.transposed(0, 2, 1, 3).reshaped([B, N, C]))
    }
}

/// `SwiGLUFFN`: w12 → chunk(2) → silu(x1) * x2 → w3.
public final class SwiGLUFFN: Module, UnaryLayer {
    @ModuleInfo var w12: Linear
    @ModuleInfo var w3: Linear

    public init(inFeatures: Int, hiddenFeatures: Int) {
        self._w12.wrappedValue = Linear(inFeatures, 2 * hiddenFeatures, bias: true)
        self._w3.wrappedValue = Linear(hiddenFeatures, inFeatures, bias: true)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x12 = w12(x)
        let parts = x12.split(parts: 2, axis: -1)
        let hidden = silu(parts[0]) * parts[1]
        return w3(hidden)
    }
}

/// `LightningDiTBlock`: adaLN-single modulation from `scale_shift_table + c0`, self-attn (RoPE) →
/// cross-attn → SwiGLU, gated residuals.
public final class LightningDiTBlock: Module {
    public let norm1: RMSNorm
    public let norm2: RMSNorm
    @ModuleInfo public var attn: Attention
    @ModuleInfo(key: "cross_attn") public var crossAttn: MultiHeadCrossAttention
    @ModuleInfo public var mlp: SwiGLUFFN
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    public init(hiddenSize: Int, numHeads: Int, mlpRatio: Int = 4, eps: Float = 1e-6) {
        norm1 = RMSNorm(dimensions: hiddenSize, eps: eps)
        norm2 = RMSNorm(dimensions: hiddenSize, eps: eps)
        self._attn.wrappedValue = Attention(dim: hiddenSize, numHeads: numHeads, eps: eps)
        self._crossAttn.wrappedValue = MultiHeadCrossAttention(dModel: hiddenSize, numHeads: numHeads, eps: eps)
        let mlpHidden = hiddenSize * mlpRatio
        self._mlp.wrappedValue = SwiGLUFFN(inFeatures: hiddenSize, hiddenFeatures: Int(2.0 / 3.0 * Double(mlpHidden)))
        self._scaleShiftTable.wrappedValue = MLXArray.zeros([6, hiddenSize])
        super.init()
    }

    /// x [B, N, D]; c0 [B, 6·D]; z [B, Nz, D] (already projected); rope tables [N, hd].
    public func callAsFunction(_ xIn: MLXArray, c0: MLXArray, z: MLXArray,
                               rope: (cos: MLXArray, sin: MLXArray)?) -> MLXArray {
        let B = xIn.dim(0)
        let mod = (scaleShiftTable.expandedDimensions(axis: 0) + c0.reshaped([B, 6, -1])).split(parts: 6, axis: 1)
        let (shiftMsa, scaleMsa, gateMsa, shiftMlp, scaleMlp, gateMlp) = (mod[0], mod[1], mod[2], mod[3], mod[4], mod[5])
        var x = xIn + gateMsa * attn(modulateAdasin(norm1(xIn), shift: shiftMsa, scale: scaleMsa), rope: rope)
        x = x + crossAttn(x, cond: z)
        x = x + gateMlp * mlp(modulateAdasin(norm2(x), shift: shiftMlp, scale: scaleMlp))
        return x
    }
}

/// `FinalLayer`: adaLN (SiLU → Linear(D → 2D) on the raw t-embedding) → RMSNorm → linear(D → p²·C).
public final class FinalLayer: Module {
    @ModuleInfo(key: "norm_final") public var normFinal: RMSNorm
    @ModuleInfo public var linear: Linear
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: Sequential

    public init(hiddenSize: Int, patchSize: Int, outChannels: Int, eps: Float = 1e-6) {
        self._normFinal.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: eps)
        self._linear.wrappedValue = Linear(hiddenSize, patchSize * patchSize * outChannels, bias: true)
        self._adaLNModulation.wrappedValue = Sequential(layers: [SiLU(), Linear(hiddenSize, 2 * hiddenSize, bias: true)])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, c: MLXArray) -> MLXArray {
        let ss = adaLNModulation(c).split(parts: 2, axis: 1)
        let (shift, scale) = (ss[0], ss[1])
        return linear(modulate(normFinal(x), shift: shift, scale: scale))
    }
}

/// `LightningDiT` (VOSR2 configuration by default).
public final class LightningDiT: Module {
    @ModuleInfo(key: "x_embedder") public var xEmbedder: PatchEmbed
    @ModuleInfo(key: "t_embedder") public var tEmbedder: TimestepEmbedder
    @ModuleInfo(key: "feat_rope") public var featRope: VisionRotaryEmbeddingFast
    @ModuleInfo(key: "t_block") public var tBlock: Sequential
    @ModuleInfo public var blocks: [LightningDiTBlock]
    @ModuleInfo(key: "final_layer") public var finalLayer: FinalLayer
    @ModuleInfo(key: "layer_norm") public var layerNorm: LayerNorm
    @ModuleInfo(key: "mlp_ca") public var mlpCa: Mlp

    public let inChannels: Int
    public let outChannels: Int
    public let patchSize: Int
    public let hiddenSize: Int
    public let numHeads: Int
    public let inputSize: Int

    public init(inputSize: Int = 64, patchSize: Int = 2, inChannels: Int = 32, outChannels: Int = 16,
                hiddenSize: Int = 1536, depth: Int = 36, numHeads: Int = 24, mlpRatio: Int = 4,
                zDims: Int = 1024, encdimRatio: Int = 3) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.patchSize = patchSize
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.inputSize = inputSize
        self._xEmbedder.wrappedValue = PatchEmbed(patchSize: patchSize, inChannels: inChannels, embedDim: hiddenSize)
        self._tEmbedder.wrappedValue = TimestepEmbedder(hiddenSize: hiddenSize)
        self._featRope.wrappedValue = VisionRotaryEmbeddingFast(dim: hiddenSize / numHeads / 2, ptSeqLen: inputSize / patchSize)
        self._tBlock.wrappedValue = Sequential(layers: [SiLU(), Linear(hiddenSize, 6 * hiddenSize, bias: true)])
        self._blocks.wrappedValue = (0..<depth).map { _ in LightningDiTBlock(hiddenSize: hiddenSize, numHeads: numHeads, mlpRatio: mlpRatio) }
        self._finalLayer.wrappedValue = FinalLayer(hiddenSize: hiddenSize, patchSize: patchSize, outChannels: outChannels)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: zDims, eps: 1e-5, affine: true)   // nn.LayerNorm default eps
        self._mlpCa.wrappedValue = Mlp(inFeatures: zDims, hiddenFeatures: hiddenSize * encdimRatio, outFeatures: hiddenSize, tanhGELU: true)
        super.init()
        train(false)   // C14
    }

    public var dtype: DType { xEmbedder.proj.weight.dtype }

    /// Quantize the transformer's `Linear` layers to `bits` with `groupSize` — the int8 tier measured
    /// lossless in VOSR-EVAL §10 (G4): every block (qkv/proj/cross-attn/SwiGLU), `mlp_ca` and
    /// `final_layer.linear`. The three `Sequential`-held Linears (`t_embedder.mlp`, `t_block`,
    /// `final_layer.adaLN_modulation`, ~21 M params) stay unquantized — MLXNN's `quantize` cannot replace an
    /// element of a `[UnaryLayer]` container (`UpdateError.unexpectedStructure`), and they are a rounding
    /// error of the 1.39 B. Convs, norms and the RoPE/scale-shift tables keep their loaded dtype.
    public func quantizeLinears(groupSize: Int = 64, bits: Int = 8) {
        quantize(model: self, groupSize: groupSize, bits: bits) { path, module in
            module is Linear && !path.contains("layers")
        }
        eval(self.parameters())
    }

    /// `layer_norm` → `mlp_ca` on the DINOv2 tokens — shared by every block; exposed so a tile driver can
    /// compute it once per tile.
    public func projectCondition(_ z: MLXArray) -> MLXArray {
        mlpCa(layerNorm(z.asType(dtype)))
    }

    /// `unpatchify`: (N, T, p·p·C) → (N, C, h·p, w·p).
    func unpatchify(_ x: MLXArray, h: Int, w: Int) -> MLXArray {
        let (n, p, c) = (x.dim(0), patchSize, outChannels)
        return x.reshaped([n, h, w, p, p, c]).transposed(0, 5, 1, 3, 2, 4).reshaped([n, c, h * p, w * p])
    }

    /// `forward_flexible(x, t, r=None, z=[zTokens])`: square NCHW input [B, 32, H, H] → velocity [B, 16, H, H].
    public func callAsFunction(_ xIn: MLXArray, t: MLXArray, z: MLXArray) -> MLXArray {
        let (H, W) = (xIn.dim(2), xIn.dim(3))
        precondition(H == W, "forward_flexible only supports square inputs")
        let hw = H / patchSize
        var x = xEmbedder(xIn.asType(dtype).transposed(0, 2, 3, 1))     // (N, T, D)
        let tEmb = tEmbedder(t)                                          // (N, D)
        let c = tEmb                                                     // r_embedder is None → r = 0
        let c0 = tBlock(c)
        let zz = projectCondition(z)
        let (cos, sin) = featRope.tables(for: hw, dtype: x.dtype)
        for block in blocks { x = block(x, c0: c0, z: zz, rope: (cos, sin)) }
        x = finalLayer(x, c: c)                                          // (N, T, p²·C)
        return unpatchify(x, h: hw, w: hw)
    }
}
