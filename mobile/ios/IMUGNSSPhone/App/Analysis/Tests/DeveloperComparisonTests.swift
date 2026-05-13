#if DEBUG
import XCTest
@testable import IMUGNSSPhone

final class DeveloperComparisonTests: XCTestCase {
    func testPositionAndVelocityResidualsUseEstimateMinusReference() {
        let snapshot = DeveloperComparison.makeSnapshot(
            iosPositionM: NavigationVectorNED(north: 10.0, east: 5.0, down: -1.0),
            fusedPositionM: NavigationVectorNED(north: 13.0, east: 1.0, down: 2.0),
            iosVelocityMps: NavigationVectorNED(north: 2.0, east: 0.0, down: 0.5),
            fusedVelocityMps: NavigationVectorNED(north: 5.0, east: 4.0, down: -0.5),
            iosCourseDeg: nil,
            fusedYawDeg: nil,
            iosAttitudeEulerDeg: nil,
            fusedEulerDeg: nil
        )

        XCTAssertEqual(snapshot.positionErrorM?.north, 3.0)
        XCTAssertEqual(snapshot.positionErrorM?.east, -4.0)
        XCTAssertEqual(snapshot.positionErrorM?.down, 3.0)
        XCTAssertEqual(snapshot.horizontalPositionErrorM ?? .nan, 5.0, accuracy: 1e-12)
        XCTAssertEqual(snapshot.velocityErrorMps?.north, 3.0)
        XCTAssertEqual(snapshot.velocityErrorMps?.east, 4.0)
        XCTAssertEqual(snapshot.velocityErrorMps?.down, -1.0)
        XCTAssertEqual(snapshot.groundSpeedErrorMps ?? .nan, sqrt(41.0) - 2.0, accuracy: 1e-12)
    }

    func testAngularResidualWrapsToSignedShortestDelta() {
        XCTAssertEqual(
            DeveloperComparison.angularErrorDeg(estimate: 2.0, reference: 358.0) ?? .nan,
            4.0,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            DeveloperComparison.angularErrorDeg(estimate: 358.0, reference: 2.0) ?? .nan,
            -4.0,
            accuracy: 1e-12
        )
    }
}
#endif
