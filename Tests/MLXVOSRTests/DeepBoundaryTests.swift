import Testing
import XCTest
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
@testable import MLXVOSR

/// 16 bits out (contract 1.52.0, AB-T-0233): a deep input is rounded to 8 bits for the reference prep and the output is
/// written at 16 bits. Offline — CoreGraphics, and MLX on the CPU stream for the output conversion.
struct DeepBoundaryTests {

    static func deepWords(_ w: Int, _ h: Int) -> [UInt16] {
        var words = [UInt16](repeating: 0, count: w * h * 4)
        for y in 0..<h { for x in 0..<w {
            let i = (y * w + x) * 4
            words[i] = UInt16(30_000 + x * 25); words[i + 1] = UInt16(10_000 + y * 7)
            words[i + 2] = UInt16((x * 977 + y * 131) % 65_536); words[i + 3] = UInt16.max
        } }
        return words
    }

    static func deepPNG(_ w: Int, _ h: Int) throws -> Data {
        let png = try #require(VOSRUpscalePackage.encodePNG16(deepWords(w, h), width: w, height: h))
        return png
    }

    @Test func surfaceDeclaresTheSixteenBitBoundary() {
        let surface = VOSRUpscalePackage.manifest.surfaces[0]
        #expect(surface.rawImageFormats == [.rawBGRA8, .rawRGBA16])
        #expect(surface.acceptsRawRGBA16)
    }

    @Test func deepInputsAreRecognised() throws {
        #expect(VOSRUpscalePackage.isDeep(.rawRGBA16(data: Data(count: 8 * 4), width: 2, height: 2)))
        #expect(VOSRUpscalePackage.isDeep(Image(format: .png, data: try Self.deepPNG(8, 4))))
        #expect(!VOSRUpscalePackage.isDeep(.rawBGRA8(data: Data(count: 16), width: 2, height: 2)))
        let png8 = try #require(VOSRUpscalePackage.encodePNG([UInt8](repeating: 90, count: 4 * 4 * 3), width: 4, height: 4))
        #expect(!VOSRUpscalePackage.isDeep(Image(format: .png, data: png8)))
    }

    /// The reference prep gets Desktop's rounding of the 16-bit image, `(v·255 + 32767) / 65535`, from raw (stride
    /// honoured) and from a 16-bit PNG alike.
    @Test func deepInputIsRoundedToEightBitsForThePrep() throws {
        let (w, h) = (5, 3)
        let words = Self.deepWords(w, h)
        var strided = Data(count: (w * 8 + 8) * h)
        for row in 0..<h {
            strided.replaceSubrange((row * (w * 8 + 8))..<(row * (w * 8 + 8) + w * 8),
                                    with: words[(row * w * 4)..<((row + 1) * w * 4)].withUnsafeBytes { Data($0) })
        }
        let raw = try VOSRUpscalePackage.decodeDeepToRGB8(.rawRGBA16(data: strided, width: w, height: h, bytesPerRow: w * 8 + 8))
        let viaPNG = try VOSRUpscalePackage.decodeDeepToRGB8(Image(format: .png, data: try Self.deepPNG(w, h)))
        #expect(raw.width == w && raw.height == h)
        for p in 0..<(w * h) { for c in 0..<3 {
            #expect(raw.pixels[p * 3 + c] == UInt8((UInt32(words[p * 4 + c]) * 255 + 32767) / 65535))
        } }
        #expect(viaPNG.pixels == raw.pixels)
        #expect(throws: VOSRPackageError.self) { _ = try VOSRUpscalePackage.decodeDeepToRGB8(.rawRGBA16(data: Data(count: 8), width: 2, height: 2)) }
    }

    @Test func deepDecodeTileHoldsTheDeclaredEnvelope() {
        #expect(VOSRConfiguration.deepDecodeTileSize == 768)
        #expect(VOSRConfiguration.deepDecodeTileSize < VOSRConfiguration().vaeTileSize)
    }
}

/// The float output is rounded to 16 bits (not 8), alpha opaque, cropped; a 16-bit PNG comes back 16-bit. XCTest, so the
/// CPU pin holds (the SwiftPM test bundle has no metallib).
final class DeepOutputXCTests: XCTestCase {
    func testOutputIsWrittenAtSixteenBits() throws {
        let (w, h) = (6, 4)
        let words: [UInt16] = Device.withDefaultDevice(Device(.cpu)) {
            var values = [Float](repeating: 0, count: 3 * 6 * 8)       // NCHW 1×3×6×8, cropped to 4×6
            for i in 0..<values.count { values[i] = Float(i % 97) / 48.0 - 1.0 }
            return VOSRUpscalePackage.toRGBA16(MLXArray(values, [1, 3, 6, 8]), cropWidth: w, cropHeight: h)
        }
        XCTAssertEqual(words.count, w * h * 4)
        XCTAssertTrue(stride(from: 3, to: words.count, by: 4).allSatisfy { words[$0] == UInt16.max })
        XCTAssertTrue(words.enumerated().contains { $0.offset % 4 != 3 && $0.element % 257 != 0 })
        // The first pixel: channel c of output (0, 0) is values[c · 48] mapped to [0, 1] and rounded to 16 bits.
        for c in 0..<3 {
            let v = Float((c * 48) % 97) / 48.0 - 1.0
            XCTAssertEqual(words[c], UInt16((min(1, max(0, v * 0.5 + 0.5)) * 65535).rounded()))
        }
        let png = try XCTUnwrap(VOSRUpscalePackage.encodePNG16(words, width: w, height: h))
        let cg = try XCTUnwrap(CGImageSourceCreateWithData(png as CFData, nil).flatMap { CGImageSourceCreateImageAtIndex($0, 0, nil) })
        XCTAssertEqual(cg.bitsPerComponent, 16); XCTAssertEqual(cg.width, w); XCTAssertEqual(cg.height, h)
    }
}
