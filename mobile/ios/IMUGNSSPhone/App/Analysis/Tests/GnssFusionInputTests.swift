import XCTest
@testable import IMUGNSSPhone

final class GnssFusionInputTests: XCTestCase {
    func testMissingHorizontalVelocitySuppressesGnssFusionInput() {
        XCTAssertNil(GnssFusionInput.make(
            latitudeDeg: 37.0,
            longitudeDeg: -122.0,
            altitudeM: 8.0,
            velN: nil,
            velE: 3.0,
            velD: nil,
            hAcc: 4.0,
            vAcc: 8.0,
            courseDeg: 90.0,
            speedAccuracyMps: 0.5,
            courseAccuracyDeg: 5.0
        ))

        XCTAssertNil(GnssFusionInput.make(
            latitudeDeg: 37.0,
            longitudeDeg: -122.0,
            altitudeM: 8.0,
            velN: 2.0,
            velE: nil,
            velD: nil,
            hAcc: 4.0,
            vAcc: 8.0,
            courseDeg: 90.0,
            speedAccuracyMps: 0.5,
            courseAccuracyDeg: 5.0
        ))
    }

    func testBuildsInputWithoutSubstitutingMissingHorizontalVelocity() {
        let input = GnssFusionInput.make(
            latitudeDeg: 37.0,
            longitudeDeg: -122.0,
            altitudeM: 8.0,
            velN: 0.0,
            velE: 12.0,
            velD: nil,
            hAcc: 4.0,
            vAcc: 8.0,
            courseDeg: 90.0,
            speedAccuracyMps: 0.5,
            courseAccuracyDeg: 0.0
        )

        XCTAssertEqual(input?.positionStdM.north, 4.0)
        XCTAssertEqual(input?.positionStdM.east, 4.0)
        XCTAssertEqual(input?.positionStdM.down, 8.0)
        XCTAssertEqual(input?.velocityNedMps.north, 0.0)
        XCTAssertEqual(input?.velocityNedMps.east, 12.0)
        XCTAssertEqual(input?.velocityNedMps.down, 0.0)
        XCTAssertEqual(input?.velocityStdMps.north, 0.5)
        XCTAssertEqual(input?.velocityStdMps.east, 0.5)
        XCTAssertEqual(input?.velocityStdMps.down, 2.5)
        XCTAssertEqual(input?.headingRad ?? .nan, .pi / 2.0, accuracy: 1e-12)
    }

    func testInvalidCoordinateSuppressesGnssFusionInput() {
        XCTAssertNil(GnssFusionInput.make(
            latitudeDeg: 120.0,
            longitudeDeg: -122.0,
            altitudeM: 8.0,
            velN: 0.0,
            velE: 12.0,
            velD: nil,
            hAcc: 4.0,
            vAcc: 8.0,
            courseDeg: 90.0,
            speedAccuracyMps: 0.5,
            courseAccuracyDeg: 5.0
        ))
    }

    func testPoorCourseAccuracySuppressesMovingFusionInput() {
        XCTAssertNil(GnssFusionInput.make(
            latitudeDeg: 37.0,
            longitudeDeg: -122.0,
            altitudeM: 8.0,
            velN: 1.0,
            velE: -1.0,
            velD: nil,
            hAcc: 4.0,
            vAcc: 8.0,
            courseDeg: 303.0,
            speedAccuracyMps: 0.31,
            courseAccuracyDeg: 180.0
        ))
    }

    func testCourseAccuracyDoesNotInflateHorizontalVelocityStdWhenUsable() {
        let input = GnssFusionInput.make(
            latitudeDeg: 37.0,
            longitudeDeg: -122.0,
            altitudeM: 8.0,
            velN: 1.0,
            velE: -1.0,
            velD: nil,
            hAcc: 4.0,
            vAcc: 8.0,
            courseDeg: 303.0,
            speedAccuracyMps: 0.31,
            courseAccuracyDeg: 30.0
        )

        XCTAssertEqual(input?.velocityStdMps.north ?? .nan, 0.31, accuracy: 1e-12)
        XCTAssertEqual(input?.velocityStdMps.east ?? .nan, 0.31, accuracy: 1e-12)
        XCTAssertNotNil(input?.headingRad)
    }
}
