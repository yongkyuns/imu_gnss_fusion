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
            speedAccuracyMps: 0.5
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
            speedAccuracyMps: 0.5
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
            speedAccuracyMps: 0.5
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
            speedAccuracyMps: 0.5
        ))
    }
}
