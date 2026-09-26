// Resample.swift — the two bicubic resamplers the reference pipeline uses, reproduced exactly.
//
//   • PIL `Image.resize(..., Image.BICUBIC)` on 8-bit RGB — the LR → HR upsample in
//     inference_vosr_onestep.py `main` (`raw_img.resize((w*u, h*u), Image.BICUBIC)`). Keys cubic with
//     a = −0.5, window clamped to the image and renormalised, fixed-point 22-bit coefficients, and an
//     8-bit quantisation between the horizontal and vertical passes. Bit-exact to Pillow's Resample.c.
//   • torch `F.interpolate(x, 448, mode="bicubic")` on float — `preprocess_raw_image` before DINOv2.
//     align_corners=False, a = −0.75, replicate-clamped taps, no antialias. Built as two interpolation
//     matrices and applied with matmul (separable, linear).
//
// Both are pure Swift/MLX (no Apple imaging framework) so the engine wrapper and the CLI share them.

import Foundation
import MLX

// MARK: - Pillow bicubic (8-bit, fixed point)

/// Bit-exact port of Pillow's `ImagingResample` for `Image.BICUBIC` on an 8-bit interleaved image.
public enum PILBicubic {
    static let precisionBits = 22                                    // 32 - 8 - 2
    static let support = 2.0                                         // bicubic filter support

    /// Pillow `bicubic_filter` (a = −0.5).
    @inline(__always) static func filter(_ xIn: Double) -> Double {
        let a = -0.5
        let x = xIn < 0 ? -xIn : xIn
        if x < 1.0 { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1 }
        if x < 2.0 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0.0
    }

    /// `precompute_coeffs` + `normalize_coeffs_8bpc` for one axis.
    static func coefficients(inSize: Int, outSize: Int) -> (bounds: [(min: Int, count: Int)], k: [Int32], ksize: Int) {
        var filterscale = Double(inSize) / Double(outSize)
        let scale = filterscale
        if filterscale < 1.0 { filterscale = 1.0 }
        let supp = support * filterscale
        let ksize = Int(ceil(supp)) * 2 + 1
        var bounds = [(min: Int, count: Int)](repeating: (0, 0), count: outSize)
        var kk = [Int32](repeating: 0, count: outSize * ksize)
        let one = Double(1 << precisionBits)
        for xx in 0..<outSize {
            let center = (Double(xx) + 0.5) * scale
            var ww = 0.0
            let ss = 1.0 / filterscale
            var xmin = Int(center - supp + 0.5); if xmin < 0 { xmin = 0 }
            var xmax = Int(center + supp + 0.5); if xmax > inSize { xmax = inSize }
            xmax -= xmin
            var k = [Double](repeating: 0, count: ksize)
            for x in 0..<xmax { let w = filter((Double(x + xmin) - center + 0.5) * ss); k[x] = w; ww += w }
            if ww != 0.0 { for x in 0..<xmax { k[x] /= ww } }
            for x in 0..<ksize {
                let v = k[x] * one
                kk[xx * ksize + x] = Int32(v < 0 ? -0.5 + v : 0.5 + v)
            }
            bounds[xx] = (xmin, xmax)
        }
        return (bounds, kk, ksize)
    }

    @inline(__always) static func clip8(_ v: Int32) -> UInt8 {
        let s = v >> Int32(precisionBits)
        return s < 0 ? 0 : (s > 255 ? 255 : UInt8(s))
    }

    /// Resize an interleaved 8-bit image with `channels` samples per pixel (Pillow does RGB as 4-byte
    /// RGBX; the per-channel arithmetic is identical). Horizontal pass first, then vertical, with 8-bit
    /// quantisation in between — exactly Pillow's `ImagingResampleInner`.
    public static func resize(_ src: [UInt8], width: Int, height: Int, channels: Int,
                              toWidth outW: Int, toHeight outH: Int) -> [UInt8] {
        precondition(src.count == width * height * channels)
        var cur = src, curW = width, curH = height
        if outW != width {
            let (bounds, kk, ksize) = coefficients(inSize: width, outSize: outW)
            var out = [UInt8](repeating: 0, count: outW * curH * channels)
            let rounding = Int32(1 << (precisionBits - 1))
            for y in 0..<curH {
                let row = y * curW * channels
                for xx in 0..<outW {
                    let (xmin, cnt) = bounds[xx]
                    for c in 0..<channels {
                        var ss = rounding
                        for x in 0..<cnt { ss += Int32(cur[row + (x + xmin) * channels + c]) * kk[xx * ksize + x] }
                        out[(y * outW + xx) * channels + c] = clip8(ss)
                    }
                }
            }
            cur = out; curW = outW
        }
        if outH != height {
            let (bounds, kk, ksize) = coefficients(inSize: height, outSize: outH)
            var out = [UInt8](repeating: 0, count: curW * outH * channels)
            let rounding = Int32(1 << (precisionBits - 1))
            for yy in 0..<outH {
                let (ymin, cnt) = bounds[yy]
                for x in 0..<curW {
                    for c in 0..<channels {
                        var ss = rounding
                        for y in 0..<cnt { ss += Int32(cur[((y + ymin) * curW + x) * channels + c]) * kk[yy * ksize + y] }
                        out[(yy * curW + x) * channels + c] = clip8(ss)
                    }
                }
            }
            cur = out; curH = outH
        }
        return cur
    }
}

// MARK: - torch bicubic (float, align_corners=False, a = −0.75, no antialias)

public enum TorchBicubic {
    @inline(__always) static func cc1(_ x: Double, _ A: Double) -> Double { ((A + 2) * x - (A + 3)) * x * x + 1 }
    @inline(__always) static func cc2(_ x: Double, _ A: Double) -> Double { ((A * x - 5 * A) * x + 8 * A) * x - 4 * A }

    /// The [outSize, inSize] interpolation matrix of `upsample_bicubic2d` along one axis.
    public static func matrix(inSize: Int, outSize: Int) -> MLXArray {
        let A = -0.75
        let scale = Double(inSize) / Double(outSize)                 // area_pixel_compute_scale, no scale_factor
        var m = [Float](repeating: 0, count: outSize * inSize)
        for o in 0..<outSize {
            let real = scale * (Double(o) + 0.5) - 0.5               // cubic: not clamped at 0
            let idx = Int(floor(real))
            let t = real - Double(idx)
            let coeffs = [cc2(t + 1.0, A), cc1(t, A), cc1(1.0 - t, A), cc2(2.0 - t, A)]
            for k in 0..<4 {
                let src = min(max(idx - 1 + k, 0), inSize - 1)        // upsample_get_value_bounded
                m[o * inSize + src] += Float(coeffs[k])
            }
        }
        return MLXArray(m, [outSize, inSize])
    }

    /// `F.interpolate(x, (outH, outW), mode="bicubic")` for NCHW `x` (any leading dims).
    public static func interpolate(_ x: MLXArray, toHeight outH: Int, toWidth outW: Int) -> MLXArray {
        let inH = x.dim(-2), inW = x.dim(-1)
        var y = x
        if outH != inH { y = matmul(matrix(inSize: inH, outSize: outH).asType(x.dtype), y) }
        if outW != inW { y = matmul(y, matrix(inSize: inW, outSize: outW).asType(x.dtype).transposed()) }
        return y
    }
}
