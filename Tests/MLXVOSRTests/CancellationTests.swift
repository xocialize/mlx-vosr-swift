//
//  CancellationTests.swift — the offline CAN gate (engine ≥ 0.27.0): CAN-1/CAN-2 pre-cancelled run()
//  propagation + classification, and CAN-3 the checkpoint-cadence declaration of record.
//

import XCTest
import Foundation
import MLXToolKit
import MLXServeConformance
@testable import MLXVOSR

final class CancellationTests: XCTestCase {

    func testCANGatePreCancelledRun() async {
        // Construction is cheap (C13) and the entry checkpoint throws before validation or decode.
        let package = VOSRUpscalePackage(configuration: VOSRConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: ImageUpscaleRequest(image: Image(format: .png, data: Data())))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclaration() {
        // peakActivationBytes ≥ 2 GB ⇒ long-run implied; the sub-second exemption is not available.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: VOSRUpscalePackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: VOSRUpscalePackage.manifest,
            posture: .cadence([
                // `VOSR2Pipeline.checkpoint` runs before every 512² DiT tile and before the decode
                // ("chunk" = one latent tile); `onTile` reports RunProgress(.upsample, step, total).
                .init(phase: .upsample, unit: .chunk, reportsRunProgress: true),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
