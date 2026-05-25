import Foundation

struct FusionStatus {
    let mountReady: Bool
    let mountReadyChanged: Bool
    let ekfInitialized: Bool
    let ekfInitializedNow: Bool
    let filterInitialized: Bool
    let filterInitializedNow: Bool
    let mountQBV: Quaternion?
}

struct FusionSnapshot {
    let positionNedM: NavigationVectorNED
    let coordinate: GeographicCoordinate?
    let velocityNedMps: (n: Double, e: Double, d: Double)
    let attitudeQNV: Quaternion
    let mountQBV: Quaternion
    let eulerRad: (roll: Double, pitch: Double, yaw: Double)
    let gyroBiasRadps: (x: Double, y: Double, z: Double)
    let accelBiasMps2: (x: Double, y: Double, z: Double)
    let initialized: Bool
    let mountReady: Bool
}

struct AlignProgressStatus {
    let isValid: Bool
    let coarseReady: Bool
    let rollSigmaDeg: Double?
    let pitchSigmaDeg: Double?
    let yawSigmaDeg: Double?
}

struct FusionResult {
    let status: FusionStatus
    let snapshot: FusionSnapshot?
    let alignProgress: AlignProgressStatus
}

struct RoadEventDetection: Equatable, Sendable {
    enum Kind: UInt32, Equatable, Sendable {
        case harshAcceleration = 1
        case harshBraking = 2
        case harshCornering = 3
        case reverse = 4
        case speedBump = 5
        case uphill = 6
        case downhill = 7
    }

    let kind: Kind
    let tSec: Double
    let startTSec: Double
    let endTSec: Double
    let durationSec: Double
    let value: Double
    let confidence: Double
}

struct TripEventCounts: Equatable, Sendable {
    var speedBumps: UInt32 = 0
    var uphill: UInt32 = 0
    var downhill: UInt32 = 0
    var reverse: UInt32 = 0
    var harshAcceleration: UInt32 = 0
    var harshBraking: UInt32 = 0
    var harshCornering: UInt32 = 0

    var harshTotal: UInt32 {
        harshAcceleration + harshBraking + harshCornering
    }
}

struct TripStatsSummary: Equatable, Sendable {
    var sampleCount: UInt32
    var invalidSampleCount: UInt32
    var dataGapCount: UInt32
    var maxSampleGapSec: Double
    var totalGapDurationSec: Double
    var durationSec: Double
    var movingDurationSec: Double
    var stationaryDurationSec: Double
    var distanceM: Double
    var reverseDurationSec: Double
    var reverseDistanceM: Double
    var uphillDistanceM: Double
    var downhillDistanceM: Double
    var elevationGainM: Double
    var elevationLossM: Double
    var meanSpeedMps: Double
    var movingMeanSpeedMps: Double
    var peakSpeedMps: Double
    var peakAccelMps2: Double
    var peakDecelMps2: Double
    var peakLateralAccelMps2: Double
    var rollingSpeedMps: Double
    var rollingAbsLongitudinalAccelMps2: Double
    var rollingAbsLateralAccelMps2: Double
    var events: TripEventCounts
    var speedBumpsPerKm: Double
    var harshEventsPerKm: Double
    var reverseSecondsPerKm: Double

    static let empty = TripStatsSummary(
        sampleCount: 0,
        invalidSampleCount: 0,
        dataGapCount: 0,
        maxSampleGapSec: 0.0,
        totalGapDurationSec: 0.0,
        durationSec: 0.0,
        movingDurationSec: 0.0,
        stationaryDurationSec: 0.0,
        distanceM: 0.0,
        reverseDurationSec: 0.0,
        reverseDistanceM: 0.0,
        uphillDistanceM: 0.0,
        downhillDistanceM: 0.0,
        elevationGainM: 0.0,
        elevationLossM: 0.0,
        meanSpeedMps: 0.0,
        movingMeanSpeedMps: 0.0,
        peakSpeedMps: 0.0,
        peakAccelMps2: 0.0,
        peakDecelMps2: 0.0,
        peakLateralAccelMps2: 0.0,
        rollingSpeedMps: 0.0,
        rollingAbsLongitudinalAccelMps2: 0.0,
        rollingAbsLateralAccelMps2: 0.0,
        events: TripEventCounts(),
        speedBumpsPerKm: 0.0,
        harshEventsPerKm: 0.0,
        reverseSecondsPerKm: 0.0
    )
}

final class FusionEngine {
    private var handle: OpaquePointer?
    private var streamEpoch: Date?
    private var harshBehaviorPreset: HarshBehaviorPreset = .balanced

    init() {
        handle = sensor_fusion_create_ekf_auto()
        applyHarshBehaviorPreset()
    }

    deinit {
        if let handle {
            sensor_fusion_destroy(handle)
        }
    }

    func resetEkfManualIdentity() {
        resetEkfManual(qBV: .identity)
    }

    func resetEkfManual(qBV: Quaternion) {
        let q = qBV.normalized
        streamEpoch = nil
        if let handle {
            sensor_fusion_reset_ekf_manual(handle, Float(q.w), Float(q.x), Float(q.y), Float(q.z))
        } else {
            handle = sensor_fusion_create_ekf_manual(Float(q.w), Float(q.x), Float(q.y), Float(q.z))
        }
        applyHarshBehaviorPreset()
    }

    func resetEkfAuto() {
        streamEpoch = nil
        if let handle {
            sensor_fusion_reset_ekf_auto(handle)
        } else {
            handle = sensor_fusion_create_ekf_auto()
        }
        applyHarshBehaviorPreset()
    }

    func setHarshBehaviorPreset(_ preset: HarshBehaviorPreset) {
        harshBehaviorPreset = preset
        applyHarshBehaviorPreset()
    }

    private func applyHarshBehaviorPreset() {
        guard let handle else { return }
        _ = sensor_fusion_set_harsh_behavior_preset(handle, harshBehaviorPreset.rawValue)
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
        return FusionResult(
            status: Self.status(from: update),
            snapshot: snapshot(),
            alignProgress: alignProgress()
        )
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
        return FusionResult(
            status: Self.status(from: update),
            snapshot: snapshot(),
            alignProgress: alignProgress()
        )
    }

    func status() -> FusionStatus? {
        guard let handle else { return nil }
        return Self.status(from: sensor_fusion_snapshot_status(handle))
    }

    func snapshot() -> FusionSnapshot? {
        guard let handle else { return nil }
        var raw = SensorFusionFfiEkfSnapshot()
        guard sensor_fusion_snapshot_ekf(handle, &raw) else { return nil }
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
            mountQBV: Quaternion(
                w: Double(raw.q_bv0),
                x: Double(raw.q_bv1),
                y: Double(raw.q_bv2),
                z: Double(raw.q_bv3)
            ),
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

    func alignProgress() -> AlignProgressStatus {
        guard let handle else {
            return AlignProgressStatus(isValid: false, coarseReady: false, rollSigmaDeg: nil, pitchSigmaDeg: nil, yawSigmaDeg: nil)
        }
        var raw = SensorFusionFfiAlignProgress()
        guard sensor_fusion_snapshot_align_progress(handle, &raw), raw.valid else {
            return AlignProgressStatus(isValid: false, coarseReady: false, rollSigmaDeg: nil, pitchSigmaDeg: nil, yawSigmaDeg: nil)
        }
        return AlignProgressStatus(
            isValid: true,
            coarseReady: raw.coarse_ready,
            rollSigmaDeg: Double(raw.roll_sigma_deg),
            pitchSigmaDeg: Double(raw.pitch_sigma_deg),
            yawSigmaDeg: Double(raw.yaw_sigma_deg)
        )
    }

    func processRoadEventMotion(
        tSec: Double,
        forwardVelocityMps: Double,
        groundSpeedMps: Double,
        longitudinalAccelMps2: Double?,
        yawRateRadps: Double?,
        pitchDeg: Double?,
        lateralAccelMps2: Double?,
        verticalAccelerationMps2: Double?
    ) -> [RoadEventDetection] {
        guard let handle else { return [] }
        var rawEvents = Array(repeating: SensorFusionFfiRoadEvent(), count: 8)
        let count = rawEvents.withUnsafeMutableBufferPointer { buffer in
            sensor_fusion_process_road_event_motion(
                handle,
                Float(tSec),
                Float(forwardVelocityMps),
                Float(groundSpeedMps),
                Float(longitudinalAccelMps2 ?? 0.0),
                longitudinalAccelMps2 != nil,
                Float(yawRateRadps ?? 0.0),
                yawRateRadps != nil,
                Float(pitchDeg ?? 0.0),
                pitchDeg != nil,
                Float(lateralAccelMps2 ?? 0.0),
                lateralAccelMps2 != nil,
                Float(verticalAccelerationMps2 ?? 0.0),
                verticalAccelerationMps2 != nil,
                buffer.baseAddress,
                UInt(buffer.count)
            )
        }
        guard count > 0 else { return [] }
        return rawEvents.prefix(Int(count)).compactMap(Self.roadEvent(from:))
    }

    func tripSummary() -> TripStatsSummary? {
        guard let handle else { return nil }
        var raw = SensorFusionFfiTripSummary()
        guard sensor_fusion_snapshot_trip_summary(handle, &raw) else { return nil }
        return TripStatsSummary(
            sampleCount: raw.sample_count,
            invalidSampleCount: raw.invalid_sample_count,
            dataGapCount: raw.data_gap_count,
            maxSampleGapSec: Double(raw.max_sample_gap_s),
            totalGapDurationSec: Double(raw.total_gap_duration_s),
            durationSec: Double(raw.duration_s),
            movingDurationSec: Double(raw.moving_duration_s),
            stationaryDurationSec: Double(raw.stationary_duration_s),
            distanceM: Double(raw.distance_m),
            reverseDurationSec: Double(raw.reverse_duration_s),
            reverseDistanceM: Double(raw.reverse_distance_m),
            uphillDistanceM: Double(raw.uphill_distance_m),
            downhillDistanceM: Double(raw.downhill_distance_m),
            elevationGainM: Double(raw.elevation_gain_m),
            elevationLossM: Double(raw.elevation_loss_m),
            meanSpeedMps: Double(raw.mean_speed_mps),
            movingMeanSpeedMps: Double(raw.moving_mean_speed_mps),
            peakSpeedMps: Double(raw.peak_speed_mps),
            peakAccelMps2: Double(raw.peak_accel_mps2),
            peakDecelMps2: Double(raw.peak_decel_mps2),
            peakLateralAccelMps2: Double(raw.peak_lateral_accel_mps2),
            rollingSpeedMps: Double(raw.rolling_speed_mps),
            rollingAbsLongitudinalAccelMps2: Double(raw.rolling_abs_longitudinal_accel_mps2),
            rollingAbsLateralAccelMps2: Double(raw.rolling_abs_lateral_accel_mps2),
            events: TripEventCounts(
                speedBumps: raw.speed_bumps,
                uphill: raw.uphill_events,
                downhill: raw.downhill_events,
                reverse: raw.reverse_events,
                harshAcceleration: raw.harsh_acceleration_events,
                harshBraking: raw.harsh_braking_events,
                harshCornering: raw.harsh_cornering_events
            ),
            speedBumpsPerKm: Double(raw.speed_bumps_per_km),
            harshEventsPerKm: Double(raw.harsh_events_per_km),
            reverseSecondsPerKm: Double(raw.reverse_seconds_per_km)
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
            ekfInitialized: raw.ekf_initialized,
            ekfInitializedNow: raw.ekf_initialized_now,
            filterInitialized: raw.filter_initialized,
            filterInitializedNow: raw.filter_initialized_now,
            mountQBV: mountQBV
        )
    }

    private static func roadEvent(from raw: SensorFusionFfiRoadEvent) -> RoadEventDetection? {
        guard let kind = RoadEventDetection.Kind(rawValue: raw.kind) else { return nil }
        return RoadEventDetection(
            kind: kind,
            tSec: Double(raw.t_s),
            startTSec: Double(raw.start_t_s),
            endTSec: Double(raw.end_t_s),
            durationSec: max(0.0, Double(raw.duration_s)),
            value: Double(raw.value),
            confidence: min(max(Double(raw.confidence), 0.0), 1.0)
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
