//
//  MaterializationTests.swift — the offline MAT gate (engine ≥ 0.19.0, contract 1.24 engine-executed
//  materialization) for a NETWORK-sourced package: MAT-1 store-stampable, MAT-2 sources declared, MAT-3
//  role/repo hygiene, MAT-4 fresh-machine posture (nil store ⇒ the lane is missing), MAT-5 explicit paths
//  satisfy — per selectable lane, because the declaration changes with `quant`.
//

import XCTest
import Foundation
import MLXToolKit
import MLXServeCore
import MLXServeConformance
@testable import MLXVOSR

final class MaterializationTests: XCTestCase {

    private func satisfiedDirectory(for quant: Quant) throws -> URL {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("vosr-mat-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        for f in VOSRConfiguration.files(for: quant) {
            FileManager.default.createFile(atPath: tmp.appendingPathComponent(f).path, contents: Data([0]))
        }
        return tmp
    }

    func testFullMATGatePassesPerLane() throws {
        for quant in [Quant.fp16, .int8, .fp32] {
            let dir = try satisfiedDirectory(for: quant)
            defer { try? FileManager.default.removeItem(at: dir) }
            let report = MaterializationConformance.check(
                freshConfiguration: VOSRConfiguration(quant: quant),
                satisfiedConfiguration: VOSRConfiguration(quant: quant, weightsDirectory: dir))
            XCTAssertTrue(report.passed, "\(quant):\n\(report.summary)")
        }
    }

    /// The store's flat layout (contract 1.24: files directly under `models--mlx-community--VOSR2-fp16/`)
    /// satisfies the lane — and only when EVERY declared file is there.
    func testStoreFlatLayoutSatisfiesTheLane() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("vosr-store-\(UUID().uuidString)")
        let repoDir = root.appendingPathComponent(ModelStore.repoFolderName(for: VOSRConfiguration.fp16Repo))
        try FileManager.default.createDirectory(at: repoDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = VOSRConfiguration(quant: .fp16, modelsRootDirectory: root)
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 1, "empty repo dir ⇒ missing")
        for f in VOSRConfiguration.files(for: .fp16).dropLast() {
            FileManager.default.createFile(atPath: repoDir.appendingPathComponent(f).path, contents: Data([0]))
        }
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 1, "one file short ⇒ still missing")
        FileManager.default.createFile(atPath: repoDir.appendingPathComponent("config.json").path, contents: Data([0]))
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: root).isEmpty)
        XCTAssertEqual(cfg.resolvedWeightsDirectory(storeRoot: root)?.standardizedFileURL.path, repoDir.standardizedFileURL.path)
        XCTAssertEqual(cfg.prewarmPaths.count, 4)
    }

    /// Register is offline — a fresh registration with no store reports a needed download.
    func testEngineNeedsDownloadOnAFreshRegistration() async throws {
        let engine = MLXServeEngine()
        _ = try await engine.register(VOSRUpscalePackage.registration, configuration: VOSRConfiguration())
        let needs = await engine.needsDownload(.imageUpscale)
        XCTAssertTrue(needs, "nothing materialized, the lane must read as needing a download")
    }
}
