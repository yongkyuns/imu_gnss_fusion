import XCTest
@testable import IMUGNSSPhone

final class RawImuSynchronizerTests: XCTestCase {
    func testCoreMotionRawAccelerationConvertsToSpecificForce() {
        let appleRaw = RawImuSynchronizer.VectorSample(timestampSec: 1.0, x: 1.0, y: -2.0, z: 9.81)

        let specificForce = CoreMotionImuConvention.specificForceFromAppleRawAcceleration(appleRaw)

        XCTAssertEqual(specificForce.timestampSec, 1.0)
        XCTAssertEqual(specificForce.x, -1.0)
        XCTAssertEqual(specificForce.y, 2.0)
        XCTAssertEqual(specificForce.z, -9.81)
    }

    func testRecordedSpecificForceFramesPassThroughOnReplay() {
        let accel = CoreMotionImuConvention.specificForceComponents(
            accelXMps2: -1.0,
            accelYMps2: 2.0,
            accelZMps2: -9.81,
            attitudeReferenceFrame: CoreMotionImuConvention.sensorFusionSpecificForceFrame
        )

        XCTAssertEqual(accel?.x, -1.0)
        XCTAssertEqual(accel?.y, 2.0)
        XCTAssertEqual(accel?.z, -9.81)
    }

    func testUnsupportedRecordedFramesAreRejected() {
        let accel = CoreMotionImuConvention.specificForceComponents(
            accelXMps2: 1.0,
            accelYMps2: 2.0,
            accelZMps2: 3.0,
            attitudeReferenceFrame: "rawAccelerometerGyro"
        )

        XCTAssertNil(accel)
    }

    func testEmitsOneSampleWhenAccelAndGyroAreBothAvailable() throws {
        var synchronizer = RawImuSynchronizer(maxSkewSec: 0.02)

        XCTAssertNil(synchronizer.pushAccel(.init(timestampSec: 10.00, x: 1.0, y: 2.0, z: 3.0)))
        let fused = synchronizer.pushGyro(.init(timestampSec: 10.01, x: 0.1, y: 0.2, z: 0.3))

        let unwrapped = try XCTUnwrap(fused)
        XCTAssertEqual(unwrapped.timestampSec, 10.01, accuracy: 1e-12)
        XCTAssertEqual(unwrapped.accel.x, 1.0)
        XCTAssertEqual(unwrapped.gyro.z, 0.3)
        XCTAssertEqual(unwrapped.skewSec, 0.01, accuracy: 1e-12)
    }

    func testDoesNotEmitDuplicatePairsUntilBothStreamsAdvance() {
        var synchronizer = RawImuSynchronizer(maxSkewSec: 0.02)

        XCTAssertNil(synchronizer.pushAccel(.init(timestampSec: 1.00, x: 1.0, y: 0.0, z: 0.0)))
        XCTAssertNotNil(synchronizer.pushGyro(.init(timestampSec: 1.00, x: 0.0, y: 1.0, z: 0.0)))
        XCTAssertNil(synchronizer.pushAccel(.init(timestampSec: 1.01, x: 2.0, y: 0.0, z: 0.0)))
        XCTAssertNotNil(synchronizer.pushGyro(.init(timestampSec: 1.01, x: 0.0, y: 2.0, z: 0.0)))
    }

    func testRejectsPairsOutsideSkewLimit() {
        var synchronizer = RawImuSynchronizer(maxSkewSec: 0.005)

        XCTAssertNil(synchronizer.pushAccel(.init(timestampSec: 2.000, x: 1.0, y: 0.0, z: 0.0)))
        XCTAssertNil(synchronizer.pushGyro(.init(timestampSec: 2.010, x: 0.0, y: 1.0, z: 0.0)))
        XCTAssertNotNil(synchronizer.pushAccel(.init(timestampSec: 2.011, x: 2.0, y: 0.0, z: 0.0)))
    }

    func testResetAllowsFreshPairing() {
        var synchronizer = RawImuSynchronizer(maxSkewSec: 0.02)

        XCTAssertNil(synchronizer.pushAccel(.init(timestampSec: 3.0, x: 1.0, y: 0.0, z: 0.0)))
        XCTAssertNotNil(synchronizer.pushGyro(.init(timestampSec: 3.0, x: 0.0, y: 1.0, z: 0.0)))

        synchronizer.reset()

        XCTAssertNil(synchronizer.pushGyro(.init(timestampSec: 3.0, x: 0.0, y: 2.0, z: 0.0)))
        XCTAssertNotNil(synchronizer.pushAccel(.init(timestampSec: 3.0, x: 2.0, y: 0.0, z: 0.0)))
    }
}
