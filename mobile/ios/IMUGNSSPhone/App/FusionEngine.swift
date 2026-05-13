import Foundation

struct FusionStatus {
    let mountReady: Bool
    let mountReadyChanged: Bool
    let reducedInitialized: Bool
    let reducedInitializedNow: Bool
    let filterInitialized: Bool
    let filterInitializedNow: Bool
    let mountQBV: Quaternion?
}

struct FusionSnapshot {
    let positionNedM: NavigationVectorNED
    let coordinate: GeographicCoordinate?
    let velocityNedMps: (n: Double, e: Double, d: Double)
    let attitudeQNV: Quaternion
    let eulerRad: (roll: Double, pitch: Double, yaw: Double)
    let gyroBiasRadps: (x: Double, y: Double, z: Double)
    let accelBiasMps2: (x: Double, y: Double, z: Double)
    let initialized: Bool
    let mountReady: Bool
}

struct FusionResult {
    let status: FusionStatus
    let snapshot: FusionSnapshot?
}

final class FusionEngine {
    private var handle: OpaquePointer?
    private var streamEpoch: Date?

    init() {
        handle = sensor_fusion_create_reduced_auto()
    }

    deinit {
        if let handle {
            sensor_fusion_destroy(handle)
        }
    }

    func resetReducedManualIdentity() {
        streamEpoch = nil
        if let handle {
            sensor_fusion_reset_reduced_manual(handle, 1.0, 0.0, 0.0, 0.0)
        } else {
            handle = sensor_fusion_create_reduced_manual(1.0, 0.0, 0.0, 0.0)
        }
    }

    func resetReducedAuto() {
        streamEpoch = nil
        if let handle {
            sensor_fusion_reset_reduced_auto(handle)
        } else {
            handle = sensor_fusion_create_reduced_auto()
        }
    }

    func processImu(
        sampleDate: Date,
        accelMps2: (x: Double, y: Double, z: Double),
        gyroRadps: (x: Double, y: Double, z: Double)
    ) -> FusionResult? {
        guard let handle else { return nil }
        let update = sensor_fusion_process_imu(
            handle,
            normalizedTimestamp(for: sampleDate),
            Float(accelMps2.x),
            Float(accelMps2.y),
            Float(accelMps2.z),
            Float(gyroRadps.x),
            Float(gyroRadps.y),
            Float(gyroRadps.z)
        )
        return FusionResult(status: Self.status(from: update), snapshot: snapshot())
    }

    func processGnss(
        sampleDate: Date,
        latitudeDeg: Double,
        longitudeDeg: Double,
        altitudeM: Double,
        positionStdM: (n: Double, e: Double, d: Double),
        velocityNedMps: (n: Double, e: Double, d: Double),
        velocityStdMps: (n: Double, e: Double, d: Double),
        headingRad: Double?
    ) -> FusionResult? {
        guard let handle else { return nil }
        let update = sensor_fusion_process_gnss(
            handle,
            normalizedTimestamp(for: sampleDate),
            latitudeDeg,
            longitudeDeg,
            altitudeM,
            Float(velocityNedMps.n),
            Float(velocityNedMps.e),
            Float(velocityNedMps.d),
            Float(positionStdM.n),
            Float(positionStdM.e),
            Float(positionStdM.d),
            Float(velocityStdMps.n),
            Float(velocityStdMps.e),
            Float(velocityStdMps.d),
            Float(headingRad ?? 0.0),
            headingRad != nil
        )
        return FusionResult(status: Self.status(from: update), snapshot: snapshot())
    }

    func status() -> FusionStatus? {
        guard let handle else { return nil }
        return Self.status(from: sensor_fusion_snapshot_status(handle))
    }

    func snapshot() -> FusionSnapshot? {
        guard let handle else { return nil }
        var raw = SensorFusionFfiReducedSnapshot()
        guard sensor_fusion_snapshot_reduced(handle, &raw) else { return nil }
        let attitudeQNV = Quaternion(w: Double(raw.q0), x: Double(raw.q1), y: Double(raw.q2), z: Double(raw.q3))
        let eulerRad = Self.eulerRad(q0: raw.q0, q1: raw.q1, q2: raw.q2, q3: raw.q3)
        return FusionSnapshot(
            positionNedM: NavigationVectorNED(
                north: Double(raw.pos_n_m),
                east: Double(raw.pos_e_m),
                down: Double(raw.pos_d_m)
            ),
            coordinate: raw.position_lla_valid
                ? GeographicCoordinate(
                    latitudeDeg: raw.lat_deg,
                    longitudeDeg: raw.lon_deg,
                    altitudeM: raw.height_m
                )
                : nil,
            velocityNedMps: (
                n: Double(raw.vel_n_mps),
                e: Double(raw.vel_e_mps),
                d: Double(raw.vel_d_mps)
            ),
            attitudeQNV: attitudeQNV,
            eulerRad: eulerRad,
            gyroBiasRadps: (
                x: Double(raw.gyro_bias_x_radps),
                y: Double(raw.gyro_bias_y_radps),
                z: Double(raw.gyro_bias_z_radps)
            ),
            accelBiasMps2: (
                x: Double(raw.accel_bias_x_mps2),
                y: Double(raw.accel_bias_y_mps2),
                z: Double(raw.accel_bias_z_mps2)
            ),
            initialized: raw.initialized,
            mountReady: raw.mount_ready
        )
    }

    private func normalizedTimestamp(for sampleDate: Date) -> Float {
        if streamEpoch == nil {
            streamEpoch = sampleDate
        }
        guard let streamEpoch else { return 0.0 }
        return Float(sampleDate.timeIntervalSince(streamEpoch))
    }

    private static func status(from raw: SensorFusionFfiUpdate) -> FusionStatus {
        let mountQBV = raw.mount_q_bv_valid
            ? Quaternion(
                w: Double(raw.mount_q_bv.0),
                x: Double(raw.mount_q_bv.1),
                y: Double(raw.mount_q_bv.2),
                z: Double(raw.mount_q_bv.3)
            )
            : nil
        return FusionStatus(
            mountReady: raw.mount_ready,
            mountReadyChanged: raw.mount_ready_changed,
            reducedInitialized: raw.reduced_initialized,
            reducedInitializedNow: raw.reduced_initialized_now,
            filterInitialized: raw.filter_initialized,
            filterInitializedNow: raw.filter_initialized_now,
            mountQBV: mountQBV
        )
    }

    private static func eulerRad(q0: Float, q1: Float, q2: Float, q3: Float) -> (roll: Double, pitch: Double, yaw: Double) {
        let w = Double(q0)
        let x = Double(q1)
        let y = Double(q2)
        let z = Double(q3)

        let sinrCosp = 2.0 * (w * x + y * z)
        let cosrCosp = 1.0 - 2.0 * (x * x + y * y)
        let roll = atan2(sinrCosp, cosrCosp)

        let sinp = 2.0 * (w * y - z * x)
        let pitch: Double
        if abs(sinp) >= 1.0 {
            pitch = sinp.sign == .minus ? -.pi / 2.0 : .pi / 2.0
        } else {
            pitch = asin(sinp)
        }

        let sinyCosp = 2.0 * (w * z + x * y)
        let cosyCosp = 1.0 - 2.0 * (y * y + z * z)
        let yaw = atan2(sinyCosp, cosyCosp)
        return (roll: roll, pitch: pitch, yaw: yaw)
    }
}
