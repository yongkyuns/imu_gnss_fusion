import XCTest
@testable import IMUGNSSPhone

final class GnssVelocityResolverTests: XCTestCase {
    func testStationarySpeedDoesNotRequireCourse() {
        let velocity = GnssVelocityResolver.horizontalVelocity(
            speedMps: 0.0,
            courseDeg: nil
        )

        XCTAssertEqual(velocity?.northMps, 0.0)
        XCTAssertEqual(velocity?.eastMps, 0.0)
    }

    func testNearStationarySpeedDoesNotRequireCourse() {
        let velocity = GnssVelocityResolver.horizontalVelocity(
            speedMps: GnssVelocityResolver.stationarySpeedThresholdMps,
            courseDeg: nil
        )

        XCTAssertEqual(velocity?.northMps, 0.0)
        XCTAssertEqual(velocity?.eastMps, 0.0)
    }

    func testMovingSpeedRequiresCourse() {
        XCTAssertNil(GnssVelocityResolver.horizontalVelocity(
            speedMps: GnssVelocityResolver.stationarySpeedThresholdMps + 0.01,
            courseDeg: nil
        ))
    }

    func testMovingSpeedAndCourseProduceNedVelocity() {
        let velocity = GnssVelocityResolver.horizontalVelocity(
            speedMps: 12.0,
            courseDeg: 90.0
        )

        XCTAssertEqual(velocity?.northMps ?? .nan, 0.0, accuracy: 1e-12)
        XCTAssertEqual(velocity?.eastMps ?? .nan, 12.0, accuracy: 1e-12)
    }
}
