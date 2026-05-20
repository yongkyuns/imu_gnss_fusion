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

    func testVelocityStdIncludesCourseAccuracy() {
        let std = GnssVelocityResolver.horizontalVelocityStdMps(
            speedMps: 2.0,
            speedAccuracyMps: 0.3,
            courseAccuracyDeg: 180.0
        )

        XCTAssertEqual(std, hypot(0.3, 2.0 * .pi), accuracy: 1e-12)
    }

    func testMissingCourseAccuracyKeepsSpeedAccuracyStd() {
        let std = GnssVelocityResolver.horizontalVelocityStdMps(
            speedMps: 12.0,
            speedAccuracyMps: 0.5,
            courseAccuracyDeg: nil
        )

        XCTAssertEqual(std, 0.5, accuracy: 1e-12)
    }

    func testPoorCourseAccuracySuppressesHeading() {
        XCTAssertNil(GnssVelocityResolver.headingRad(
            courseDeg: 303.0,
            courseAccuracyDeg: 180.0
        ))
    }

    func testUsableCourseAccuracyKeepsHeading() {
        XCTAssertEqual(
            GnssVelocityResolver.headingRad(courseDeg: 90.0, courseAccuracyDeg: 10.0) ?? .nan,
            .pi / 2.0,
            accuracy: 1e-12
        )
    }
}
