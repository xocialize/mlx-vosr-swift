// Ops.swift — the small shared ops behind the three VOSR2 components (isomorphic helpers, not a
// framework). Each mirrors one upstream function; names follow the upstream file that owns them.
//
//   models/pos_embed.py       rotate_half (EVA-02, interleaved pairs)   → rotateHalfInterleaved
//   models/lightningdit.py    modulate_adasin / modulate / timestep_embedding
//   tiled_vae.py              _gaussian_weights / _make_tile_grid
//   (mlx numerics)            WinogradFreeConv2d — the fleet route around mlx's lossy Winograd conv2d

import Foundation
import MLX
import MLXNN

// MARK: - EVA rotary helpers (models/pos_embed.py)

/// `rotate_half` from EVA-02: view the last dim as (d, 2) pairs and map (x1, x2) → (−x2, x1).
/// This is the INTERLEAVED convention (pairs are adjacent elements), not the GPT-NeoX split-half one.
@inline(__always)
func rotateHalfInterleaved(_ x: MLXArray) -> MLXArray {
    let shape = x.shape
    let pairs = shape[shape.count - 1] / 2
    let xr = x.reshaped(Array(shape.dropLast()) + [pairs, 2])
    let x1 = xr[.ellipsis, 0]
    let x2 = xr[.ellipsis, 1]
    return stacked([-x2, x1], axis: -1).reshaped(shape)
}

/// `t * cos + rotate_half(t) * sin` — `VisionRotaryEmbeddingFast.forward`. `t` is [B, H, N, Dh];
/// `cos`/`sin` are [N, Dh] and broadcast over batch and heads.
@inline(__always)
func applyVisionRope(_ t: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    t * cos + rotateHalfInterleaved(t) * sin
}

/// The EVA 2-D rotary tables for an `hw × hw` patch grid, exactly `VisionRotaryEmbeddingFast.__init__`
/// / `LightningDiT._get_dynamic_rope`: `dim` = half the head dim, `ptSeqLen` = the training grid.
/// Row index = h * hw + w; the first `dim` columns carry the h angle, the last `dim` the w angle; every
/// frequency is repeated for its (x1, x2) pair. Returns (cos, sin) each [hw·hw, 2·dim] in float32.
func visionRotaryTables(hwSeqLen hw: Int, ptSeqLen pt: Int, dim: Int, theta: Float = 10000) -> (MLXArray, MLXArray) {
    let nFreq = dim / 2
    var freqs = [Float](repeating: 0, count: nFreq)
    for i in 0..<nFreq { freqs[i] = 1.0 / powf(theta, Float(2 * i) / Float(dim)) }
    // t = arange(hw) / hw * pt  (float32)
    var angles = [Float](repeating: 0, count: hw * hw * 2 * dim)
    for h in 0..<hw {
        let th = Float(h) / Float(hw) * Float(pt)
        for w in 0..<hw {
            let tw = Float(w) / Float(hw) * Float(pt)
            let base = (h * hw + w) * 2 * dim
            for f in 0..<nFreq {
                angles[base + 2 * f] = th * freqs[f]
                angles[base + 2 * f + 1] = th * freqs[f]
                angles[base + dim + 2 * f] = tw * freqs[f]
                angles[base + dim + 2 * f + 1] = tw * freqs[f]
            }
        }
    }
    let a = MLXArray(angles, [hw * hw, 2 * dim])
    return (cos(a), sin(a))
}

// MARK: - adaLN modulation (models/lightningdit.py)

/// `modulate_adasin(x, shift, scale)` with `shift` present: `x * (1 + scale) + shift`; the block passes
/// [B, 1, D] shift/scale, which broadcast over tokens.
@inline(__always)
func modulateAdasin(_ x: MLXArray, shift: MLXArray, scale: MLXArray) -> MLXArray {
    x * (1 + scale) + shift
}

/// `modulate(x, shift, scale)` (FinalLayer): shift/scale are [B, D] and are unsqueezed at dim 1.
@inline(__always)
func modulate(_ x: MLXArray, shift: MLXArray, scale: MLXArray) -> MLXArray {
    x * (1 + scale.expandedDimensions(axis: 1)) + shift.expandedDimensions(axis: 1)
}

/// `TimestepEmbedder.timestep_embedding(t, dim)` — sinusoidal, cos first then sin, float32.
func timestepEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Float = 10000) -> MLXArray {
    let half = dim / 2
    var f = [Float](repeating: 0, count: half)
    let negLog = -logf(maxPeriod)
    for i in 0..<half { f[i] = expf(negLog * Float(i) / Float(half)) }
    let freqs = MLXArray(f, [1, half])
    let args = t.asType(.float32).expandedDimensions(axis: 1) * freqs   // [B, half]
    var emb = concatenated([cos(args), sin(args)], axis: -1)
    if dim % 2 == 1 { emb = concatenated([emb, MLXArray.zeros([emb.dim(0), 1])], axis: -1) }
    return emb
}

// MARK: - tiling helpers (tiled_vae.py)

/// `_gaussian_weights(tile_h, tile_w, channels)` → [1, C, tile_h, tile_w] float32 (NCHW like the reference).
func gaussianWeights(tileH: Int, tileW: Int, channels: Int) -> MLXArray {
    let variance: Float = 0.01
    let midH = Float(tileH - 1) / 2, midW = Float(tileW - 1) / 2
    var wy = [Float](repeating: 0, count: tileH), wx = [Float](repeating: 0, count: tileW)
    for y in 0..<tileH { let d = (Float(y) - midH) / Float(tileH); wy[y] = expf(-(d * d) / (2 * variance)) }
    for x in 0..<tileW { let d = (Float(x) - midW) / Float(tileW); wx[x] = expf(-(d * d) / (2 * variance)) }
    var w = [Float](repeating: 0, count: tileH * tileW)
    for y in 0..<tileH { for x in 0..<tileW { w[y * tileW + x] = wy[y] * wx[x] } }
    let g = MLXArray(w, [1, 1, tileH, tileW])
    return broadcast(g, to: [1, channels, tileH, tileW])
}

/// `_make_tile_grid(length, tile, overlap)` — sorted, deduplicated tile starts covering `length`.
func makeTileGrid(length: Int, tile: Int, overlap: Int) -> [Int] {
    let stride = max(tile - overlap, 1)
    if length <= tile { return [0] }
    var positions = Array(Swift.stride(from: 0, through: length - tile, by: stride))
    if let last = positions.last, last + tile < length { positions.append(length - tile) }
    return Array(Set(positions)).sorted()
}

// MARK: - mlx Winograd conv2d window (fleet lesson AB-L-0163)

/// How a 3×3 conv inside mlx's Winograd F(6×6,3×3) window runs (mlx-swift ≤ 0.31.6 Metal numerics).
/// mlx dispatches Winograd iff kernel 3×3 · stride 1 · dilation 1 · groups 1 · C % 32 == 0 · O % 32 == 0 ·
/// C + O ≥ 256 · N·H·W ≥ 4096. On M5 that path runs its inner GEMM in TF32 (~6.4e-3 relL2 per conv in
/// fp32, ~5.8e-2 in bf16). conv3d with kT = 1 lands on the exact implicit-GEMM kernel.
public enum VOSRConvRoute: String, Sendable {
    /// mlx's default kernel — fastest, lossy on M5 inside the window.
    case winograd
    /// conv3d with kT = 1 — exact in every dtype, 1.3–4× slower at 256/512-channel shapes.
    case conv3d

    /// `VOSR_VAE_CONV_ROUTE` = winograd | conv3d, if set (validation lever).
    static var environmentOverride: VOSRConvRoute? {
        getenv("VOSR_VAE_CONV_ROUTE").flatMap { VOSRConvRoute(rawValue: String(cString: $0)) }
    }
}

/// A `Conv2d` whose forward takes the exact conv3d route when the input hits mlx's Winograd window.
/// Shapes outside the window always run plain conv2d (already exact). Same keys as `Conv2d`.
final class WinogradFreeConv2d: Conv2d {
    var route: VOSRConvRoute = VOSRConvRoute.environmentOverride ?? .conv3d

    /// mlx's Winograd dispatch predicate, evaluated on the actual NHWC input.
    static func takesWinograd(input x: MLXArray, weight: MLXArray, stride: (Int, Int), dilation: (Int, Int), groups: Int) -> Bool {
        guard x.ndim == 4, groups == 1, stride == (1, 1), dilation == (1, 1),
              weight.dim(1) == 3, weight.dim(2) == 3 else { return false }
        let (c, o) = (x.dim(3), weight.dim(0))
        return c % 32 == 0 && o % 32 == 0 && c + o >= 256 && x.dim(0) * x.dim(1) * x.dim(2) >= 4096
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard route == .conv3d,
              Self.takesWinograd(input: x, weight: weight, stride: stride, dilation: dilation, groups: groups)
        else { return super.callAsFunction(x) }
        var y = conv3d(x.expandedDimensions(axis: 1), weight.expandedDimensions(axis: 1),
                       stride: [1, 1, 1], padding: [0, padding.0, padding.1]).squeezed(axis: 1)
        if let bias { y = y + bias }
        return y
    }
}
