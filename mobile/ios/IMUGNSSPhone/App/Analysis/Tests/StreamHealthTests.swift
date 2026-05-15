import XCTest
@testable import IMUGNSSPhone

final class StreamHealthTests: XCTestCase {
    func testLiveStreamsAreNominal() {
        let now = Date(timeIntervalSince1970: 100.0)

        let health = StreamHealth.evaluate(
            now: now,
            lastImuSampleDate: Date(timeIntervalSince1970: 99.7),
            lastGnssSampleDate: Date(timeIntervalSince1970: 98.0),
            motionError: nil,
            locationError: nil,
            recordingError: nil,
            isRecording: true
        )

        XCTAssertEqual(health.imu, .live)
        XCTAssertEqual(health.gnss, .live)
        XCTAssertEqual(health.recording, .live)
        XCTAssertEqual(health.severity, .nominal)
    }

    func testStaleImuIsWarning() {
        let now = Date(timeIntervalSince1970: 100.0)

        let health = StreamHealth.evaluate(
            now: now,
            lastImuSampleDate: Date(timeIntervalSince1970: 98.0),
            lastGnssSampleDate: Date(timeIntervalSince1970: 99.0),
            motionError: nil,
            locationError: nil,
            recordingError: nil,
            isRecording: false
        )

        XCTAssertEqual(health.imu, .stale)
        XCTAssertEqual(health.severity, .warning)
        XCTAssertEqual(health.shortTitle, "IMU Stale")
    }

    func testErrorsAreCritical() {
        let now = Date(timeIntervalSince1970: 100.0)

        let health = StreamHealth.evaluate(
            now: now,
            lastImuSampleDate: Date(timeIntervalSince1970: 99.9),
            lastGnssSampleDate: Date(timeIntervalSince1970: 99.0),
            motionError: "motion stopped",
            locationError: nil,
            recordingError: nil,
            isRecording: true
        )

        XCTAssertEqual(health.imu, .error("motion stopped"))
        XCTAssertEqual(health.severity, .critical)
        XCTAssertEqual(health.shortTitle, "IMU Error")
    }
}
