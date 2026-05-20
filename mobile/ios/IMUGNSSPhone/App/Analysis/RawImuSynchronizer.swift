import Foundation

enum CoreMotionImuConvention {
    static let sensorFusionSpecificForceFrame = "rawAccelerometerGyroSpecificForce"

    static func specificForceFromAppleRawAcceleration(_ sample: RawImuSynchronizer.VectorSample) -> RawImuSynchronizer.VectorSample {
        RawImuSynchronizer.VectorSample(
            timestampSec: sample.timestampSec,
            x: -sample.x,
            y: -sample.y,
            z: -sample.z
        )
    }

    static func specificForceComponents(
        accelXMps2: Double,
        accelYMps2: Double,
        accelZMps2: Double,
        attitudeReferenceFrame: String?
    ) -> (x: Double, y: Double, z: Double)? {
        guard attitudeReferenceFrame == sensorFusionSpecificForceFrame else {
            return nil
        }
        return (accelXMps2, accelYMps2, accelZMps2)
    }
}

struct RawImuSynchronizer: Equatable, Sendable {
    struct VectorSample: Equatable, Sendable {
        let timestampSec: TimeInterval
        let x: Double
        let y: Double
        let z: Double
    }

    struct FusedSample: Equatable, Sendable {
        let timestampSec: TimeInterval
        let accel: VectorSample
        let gyro: VectorSample

        var skewSec: TimeInterval {
            abs(accel.timestampSec - gyro.timestampSec)
        }
    }

    var maxSkewSec: TimeInterval = 0.02

    private var latestAccel: VectorSample?
    private var latestGyro: VectorSample?
    private var lastEmittedAccelTimestampSec: TimeInterval?
    private var lastEmittedGyroTimestampSec: TimeInterval?

    init(maxSkewSec: TimeInterval = 0.02) {
        self.maxSkewSec = maxSkewSec
    }

    mutating func pushAccel(_ sample: VectorSample) -> FusedSample? {
        latestAccel = sample
        return makeFusedSampleIfReady()
    }

    mutating func pushGyro(_ sample: VectorSample) -> FusedSample? {
        latestGyro = sample
        return makeFusedSampleIfReady()
    }

    mutating func reset() {
        latestAccel = nil
        latestGyro = nil
        lastEmittedAccelTimestampSec = nil
        lastEmittedGyroTimestampSec = nil
    }

    private mutating func makeFusedSampleIfReady() -> FusedSample? {
        guard let latestAccel, let latestGyro else { return nil }
        guard abs(latestAccel.timestampSec - latestGyro.timestampSec) <= maxSkewSec else {
            return nil
        }
        if let lastEmittedAccelTimestampSec,
           latestAccel.timestampSec <= lastEmittedAccelTimestampSec {
            return nil
        }
        if let lastEmittedGyroTimestampSec,
           latestGyro.timestampSec <= lastEmittedGyroTimestampSec {
            return nil
        }

        lastEmittedAccelTimestampSec = latestAccel.timestampSec
        lastEmittedGyroTimestampSec = latestGyro.timestampSec
        return FusedSample(
            timestampSec: max(latestAccel.timestampSec, latestGyro.timestampSec),
            accel: latestAccel,
            gyro: latestGyro
        )
    }
}
