//  VOSRCoreTests.swift — structural tests for the VOSR2 core that need no weights (shapes, parameter
//  counts pinned to the checkpoint inventory, inference mode, the RoPE table formula, the tile grid and
//  Gaussian blend, the two bicubic resamplers), plus the S0 key contract when `VOSR_WEIGHTS` points at the
//  converter's output directory. Everything runs on the CPU stream (the SPM test bundle's metallib is
//  unreliable; CPU fp32 is also the strict lane).

import Foundation
import Testing
import MLX
import MLXNN
import MLXFast
import MLXRandom
@testable import VOSRMLX

func withCPU<R>(_ body: () throws -> R) rethrows -> R {
    try Device.withDefaultDevice(Device(.cpu), body)
}

let weightsDir: URL? = ProcessInfo.processInfo.environment["VOSR_WEIGHTS"].map { URL(fileURLWithPath: $0) }

@Suite("VOSR2 core — pure-Swift structure (no MLX)")
struct VOSRPureStructureTests {

    @Test("Tile grid matches tiled_vae.py (128 latent, tile 64, overlap 8 → 0/56/64)")
    func tileGrid() {
        #expect(makeTileGrid(length: 128, tile: 64, overlap: 8) == [0, 56, 64])
        #expect(makeTileGrid(length: 64, tile: 64, overlap: 8) == [0])
        #expect(makeTileGrid(length: 200, tile: 64, overlap: 8) == [0, 56, 112, 136])
    }

    @Test("PIL bicubic ×2 is a partition of unity on a flat image; coefficient rows sum to 1<<22")
    func pilBicubic() {
        let flat = [UInt8](repeating: 77, count: 8 * 8 * 3)
        let up = PILBicubic.resize(flat, width: 8, height: 8, channels: 3, toWidth: 16, toHeight: 16)
        #expect(up.allSatisfy { $0 == 77 })
        let (bounds, k, ksize) = PILBicubic.coefficients(inSize: 8, outSize: 16)
        #expect(ksize == 5)
        for xx in 0..<16 {
            let s = (0..<bounds[xx].count).reduce(Int32(0)) { $0 + k[xx * ksize + $1] }
            #expect(abs(Int(s) - (1 << 22)) <= 2)
        }
    }
}

// MARK: - MLX-touching tests run under XCTest
//
// `swift test` runs XCTest suites in a process that loaded this .xctest through NSBundle, which is how mlx's
// SwiftPM metallib lookup (`NSBundle.allBundles`) finds `mlx-swift_Cmlx.bundle`; the swift-testing runner's
// dlopen'ed bundle is invisible to it and every MLX call — CPU stream included, the scheduler builds the GPU
// stream at start-up — dies with "Failed to load the default metallib". So anything that creates an MLXArray
// lives here (the sibling packages' MAT/CAN suites are XCTest for the same reason).

import XCTest

final class VOSRStructureXCTests: XCTestCase {

    /// Parameter tensor counts match the checkpoint inventories (transformer 849 · vae 186 · dinov2 256).
    func testTensorCounts() {
        withCPU {
            XCTAssertEqual(LightningDiT().parameters().flattened().count, 849)
            XCTAssertEqual(AutoencoderKLQwenImage2D().parameters().flattened().count, 186)
            XCTAssertEqual(DINOv2ViT().parameters().flattened().count, 256)
        }
    }

    /// Born in inference mode (C14).
    func testInferenceMode() {
        withCPU {
            XCTAssertFalse(LightningDiT().training)
            XCTAssertFalse(AutoencoderKLQwenImage2D().training)
            XCTAssertFalse(DINOv2ViT().training)
        }
    }

    /// DiT shapes: [1,32,64,64] → velocity [1,16,64,64]; the dynamic RoPE at a 16×16 grid also runs.
    func testDiTShapes() {
        withCPU {
            let m = LightningDiT(depth: 2)
            let z = MLXArray.zeros([1, 1024, 1024])
            let u = m(MLXArray.zeros([1, 32, 64, 64]), t: MLXArray([Float(1)]), z: z)
            eval(u); XCTAssertEqual(u.shape, [1, 16, 64, 64])
            let u2 = m(MLXArray.zeros([1, 32, 32, 32]), t: MLXArray([Float(1)]), z: z)
            eval(u2); XCTAssertEqual(u2.shape, [1, 16, 32, 32])
        }
    }

    /// VAE shapes: 64² → latent 8², decode back; DINOv2 448² → 1024 tokens.
    func testVAEAndDINOShapes() {
        withCPU {
            let vae = AutoencoderKLQwenImage2D()
            let z = vae.encodeNormalized(MLXArray.zeros([1, 3, 64, 64])); eval(z)
            XCTAssertEqual(z.shape, [1, 16, 8, 8])
            let x = vae.decodeNormalized(z); eval(x)
            XCTAssertEqual(x.shape, [1, 3, 64, 64])
            let d = DINOv2ViT(depth: 1, featureLayer: 0)
            let f = d.features(MLXArray.zeros([1, 3, 448, 448])); eval(f)
            XCTAssertEqual(f.shape, [1, 1024, 1024])
        }
    }

    /// Vision RoPE tables: hw=32 rows are h-major, pairs share a frequency, cos²+sin² = 1.
    func testRopeTables() {
        withCPU {
            let (c, s) = visionRotaryTables(hwSeqLen: 32, ptSeqLen: 32, dim: 32)
            XCTAssertEqual(c.shape, [1024, 64])
            XCTAssertLessThan((c * c + s * s - 1).abs().max().item(Float.self), 1e-5)
            let row = c[32].asArray(Float.self)   // token (h=1, w=0)
            XCTAssertEqual(row[0], cosf(1.0), accuracy: 1e-6)
            XCTAssertEqual(row[1], cosf(1.0), accuracy: 1e-6)
            XCTAssertEqual(row[32], 1.0, accuracy: 1e-6)
        }
    }

    /// rotate_half is the interleaved-pair rotation.
    func testRotateHalf() {
        withCPU {
            let x = MLXArray([Float](arrayLiteral: 1, 2, 3, 4), [1, 4])
            XCTAssertEqual(rotateHalfInterleaved(x).asArray(Float.self), [-2, 1, -4, 3])
        }
    }

    /// The VAE mid-block's query-chunked attention equals the fused single-head SDPA (exercised by forcing
    /// a sequence above the threshold with a 3-channel toy head).
    func testChunkedAttentionMatchesFused() {
        withCPU {
            let n = AttentionBlock2D.chunkedQueryThreshold + 100
            let q = MLXRandom.normal([1, 1, n, 3], key: MLXRandom.key(1))
            let k = MLXRandom.normal([1, 1, n, 3], key: MLXRandom.key(2))
            let v = MLXRandom.normal([1, 1, n, 3], key: MLXRandom.key(3))
            let fused = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.5, mask: nil)
            let chunked = AttentionBlock2D.attention(q, k, v, scale: 0.5)
            eval(fused, chunked)
            XCTAssertLessThan((fused - chunked).abs().max().item(Float.self), 1e-5)
        }
    }

    /// Gaussian blend peaks at the centre and vanishes at the corners.
    func testGaussianWeights() {
        withCPU {
            let g = gaussianWeights(tileH: 64, tileW: 64, channels: 16)
            XCTAssertEqual(g.shape, [1, 16, 64, 64])
            XCTAssertGreaterThan(g[0, 0, 31, 31].item(Float.self), 0.99)
            XCTAssertLessThan(g[0, 0, 0, 0].item(Float.self), 1e-4)
        }
    }

    /// torch bicubic matrix rows sum to 1; 512→448 hits the reference weights at the first row.
    func testTorchBicubicMatrix() {
        withCPU {
            let m = TorchBicubic.matrix(inSize: 512, outSize: 448)
            XCTAssertEqual(m.shape, [448, 512])
            XCTAssertTrue(m.sum(axis: 1).asArray(Float.self).allSatisfy { abs($0 - 1) < 1e-5 })
            let r0 = m[0].asArray(Float.self)   // real = 512/448·0.5 − 0.5 ≈ 0.071: taps −1(→0),0,1,2
            XCTAssertEqual(r0[3], 0); XCTAssertGreaterThan(r0[0], 0.8); XCTAssertGreaterThan(r0[1], 0); XCTAssertLessThan(r0[2], 0)
        }
    }
}

/// S0 key contract against the converter's output (`VOSR_WEIGHTS=<dir>`), skipped when unset.
final class VOSRKeyContractXCTests: XCTestCase {

    func testEveryConvertedCheckpointLoadsExactly() throws {
        guard let dir = weightsDir else { throw XCTSkip("VOSR_WEIGHTS not set") }
        try withCPU {
            let vae = AutoencoderKLQwenImage2D()
            try vae.loadExact(url: dir.appendingPathComponent("vae_fp32.safetensors"), component: "vae")
            let dino = DINOv2ViT()
            try dino.loadExact(url: dir.appendingPathComponent("dinov2_fp32.safetensors"), component: "dinov2")
            let dit = LightningDiT()
            try dit.loadExact(url: dir.appendingPathComponent("transformer_fp32.safetensors"), component: "transformer")
            // The checkpoint's RoPE buffers equal the formula (the dynamic path is the same function).
            let (c, _) = visionRotaryTables(hwSeqLen: 32, ptSeqLen: 32, dim: 32)
            XCTAssertLessThan((dit.featRope.freqsCos - c).abs().max().item(Float.self), 1e-5)
        }
    }

    func testPartialCheckpointIsRefused() throws {
        guard let dir = weightsDir else { throw XCTSkip("VOSR_WEIGHTS not set") }
        try withCPU {
            var sd = try MLX.loadArrays(url: dir.appendingPathComponent("vae_fp32.safetensors"))
            sd.removeValue(forKey: "quant_conv.bias")
            XCTAssertThrowsError(try AutoencoderKLQwenImage2D().loadExact(sd, component: "vae"))
        }
    }
}

// MARK: - PIL bicubic parity against Pillow's own output (VOSR_GOLDENS_DIR)

import ImageIO
import CoreGraphics

private let goldensDir: URL? = ProcessInfo.processInfo.environment["VOSR_GOLDENS_DIR"].map { URL(fileURLWithPath: $0) }

private func rgb8(_ url: URL) -> (px: [UInt8], w: Int, h: Int)? {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil), let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
    let (w, h) = (cg.width, cg.height)
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    guard let ctx = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return nil }
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    var rgb = [UInt8](repeating: 0, count: w * h * 3)
    for i in 0..<(w * h) { rgb[i * 3] = rgba[i * 4]; rgb[i * 3 + 1] = rgba[i * 4 + 1]; rgb[i * 3 + 2] = rgba[i * 4 + 2] }
    return (rgb, w, h)
}

@Suite("PIL bicubic — bit-exact vs Pillow 12 (VOSR_GOLDENS_DIR)")
struct PILBicubicParityTests {
    @Test("×2 and ×4 of the 64×48 crop reproduce Pillow's PNGs byte for byte", .enabled(if: goldensDir != nil))
    func pillowParity() throws {
        let d = goldensDir!
        let src = try #require(rgb8(d.appendingPathComponent("pil_src_64x48.png")))
        for (scale, name) in [(2, "pil_bicubic_x2_128x96.png"), (4, "pil_bicubic_x4_256x192.png")] {
            let ref = try #require(rgb8(d.appendingPathComponent(name)))
            let got = PILBicubic.resize(src.px, width: src.w, height: src.h, channels: 3, toWidth: src.w * scale, toHeight: src.h * scale)
            #expect(got.count == ref.px.count)
            let mismatches = zip(got, ref.px).filter { $0 != $1 }.count
            #expect(mismatches == 0, "×\(scale): \(mismatches) of \(ref.px.count) samples differ")
        }
    }
}
