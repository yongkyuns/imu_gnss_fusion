import Foundation

struct GnssAccuracy: Equatable, Sendable {
    enum Quality: String, Equatable, Sendable {
        case unavailable
        case good
        case fair
        case poor
    }

    var horizontalAccuracyM: Double?
    var verticalAccuracyM: Double?
    var speedAccuracyMps: Double?

    init(
        horizontalAccuracyM: Double?,
        verticalAccuracyM: Double? = nil,
        speedAccuracyMps: Double? = nil
    ) {
        self.horizontalAccuracyM = Self.validAccuracy(horizontalAccuracyM)
        self.verticalAccuracyM = Self.validAccuracy(verticalAccuracyM)
        self.speedAccuracyMps = Self.validAccuracy(speedAccuracyMps)
    }

    func quality(
        goodHorizontalM: Double = 8.0,
        fairHorizontalM: Double = 25.0
    ) -> Quality {
        guard let horizontalAccuracyM else { return .unavailable }
        if horizontalAccuracyM <= goodHorizontalM {
            return .good
        }
        if horizontalAccuracyM <= fairHorizontalM {
            return .fair
        }
        return .poor
    }

    private static func validAccuracy(_ value: Double?) -> Double? {
        guard let value, value.isFinite, value >= 0.0 else { return nil }
        return value
    }
}

private extension StreamHealth.ChannelState {
    var blocksFusion: Bool {
        switch self {
        case .live:
            return false
        case .waiting, .stale, .unavailable, .error:
            return true
        }
    }
}

struct FusionHealth: Equatable, Sendable {
    enum State: String, Equatable, Sendable {
        case notReady
        case initializing
        case running
        case stable
        case degraded
        case degradedDeadReckoning
        case awaitingGnssReseed
    }

    var state: State
    var initialized: Bool
    var mountReady: Bool
    var gnssQuality: GnssAccuracy.Quality
    var fusedConfidence: Double
    var running: Bool
    var stable: Bool
    var degraded: Bool
    var navigationUsable: Bool
    var reasonMask: UInt32
    var postInitTimeSec: Double
    var distanceM: Double
    var tailDurationSec: Double
    var tailSamples: UInt32
    var mountTailDriftDeg: Double
    var mountTailStdDeg: Double
    var gyroBiasTailDriftRadps: Double
    var gyroBiasTailStdRadps: Double
    var accelBiasTailDriftMps2: Double
    var accelBiasTailStdMps2: Double
    var mountSigmaMaxDeg: Double
    var attitudeSigmaMaxDeg: Double

    init(
        state: State,
        initialized: Bool,
        mountReady: Bool,
        gnssQuality: GnssAccuracy.Quality,
        fusedConfidence: Double,
        running: Bool? = nil,
        stable: Bool? = nil,
        degraded: Bool? = nil,
        navigationUsable: Bool? = nil,
        reasonMask: UInt32 = 0,
        postInitTimeSec: Double = 0.0,
        distanceM: Double = 0.0,
        tailDurationSec: Double = 0.0,
        tailSamples: UInt32 = 0,
        mountTailDriftDeg: Double = 0.0,
        mountTailStdDeg: Double = 0.0,
        gyroBiasTailDriftRadps: Double = 0.0,
        gyroBiasTailStdRadps: Double = 0.0,
        accelBiasTailDriftMps2: Double = 0.0,
        accelBiasTailStdMps2: Double = 0.0,
        mountSigmaMaxDeg: Double = 0.0,
        attitudeSigmaMaxDeg: Double = 0.0
    ) {
        self.state = state
        self.initialized = initialized
        self.mountReady = mountReady
        self.gnssQuality = gnssQuality
        self.fusedConfidence = min(max(fusedConfidence, 0.0), 1.0)
        self.running = running ?? (state == .running || state == .stable)
        self.stable = stable ?? (state == .stable)
        self.degraded = degraded ?? (state == .degraded || state == .degradedDeadReckoning || state == .awaitingGnssReseed)
        self.navigationUsable = navigationUsable ?? (state == .running || state == .stable || state == .degraded || state == .degradedDeadReckoning)
        self.reasonMask = reasonMask
        self.postInitTimeSec = postInitTimeSec
        self.distanceM = distanceM
        self.tailDurationSec = tailDurationSec
        self.tailSamples = tailSamples
        self.mountTailDriftDeg = mountTailDriftDeg
        self.mountTailStdDeg = mountTailStdDeg
        self.gyroBiasTailDriftRadps = gyroBiasTailDriftRadps
        self.gyroBiasTailStdRadps = gyroBiasTailStdRadps
        self.accelBiasTailDriftMps2 = accelBiasTailDriftMps2
        self.accelBiasTailStdMps2 = accelBiasTailStdMps2
        self.mountSigmaMaxDeg = mountSigmaMaxDeg
        self.attitudeSigmaMaxDeg = attitudeSigmaMaxDeg
    }

    static let notReady = FusionHealth(
        state: .notReady,
        initialized: false,
        mountReady: false,
        gnssQuality: .unavailable,
        fusedConfidence: 0.0
    )

    static func from(
        raw: SensorFusionFfiHealth,
        initialized: Bool,
        mountReady: Bool,
        gnssAccuracy: GnssAccuracy
    ) -> FusionHealth {
        let state: State
        switch raw.state {
        case SENSOR_FUSION_STATE_INITIALIZING:
            state = .initializing
        case SENSOR_FUSION_STATE_RUNNING:
            state = .running
        case SENSOR_FUSION_STATE_STABLE:
            state = .stable
        case SENSOR_FUSION_STATE_DEGRADED:
            state = .degraded
        case SENSOR_FUSION_STATE_DEGRADED_DEAD_RECKONING:
            state = .degradedDeadReckoning
        case SENSOR_FUSION_STATE_AWAITING_GNSS_RESEED:
            state = .awaitingGnssReseed
        default:
            state = .notReady
        }
        let quality = gnssAccuracy.quality()
        return FusionHealth(
            state: state,
            initialized: initialized,
            mountReady: mountReady,
            gnssQuality: quality,
            fusedConfidence: confidence(
                state: state,
                gnssQuality: quality,
                mountReady: mountReady,
                initialized: initialized
            ),
            running: raw.running,
            stable: raw.stable,
            degraded: raw.degraded,
            navigationUsable: raw.navigation_usable,
            reasonMask: raw.reason_mask,
            postInitTimeSec: Double(raw.post_init_time_s),
            distanceM: Double(raw.distance_m),
            tailDurationSec: Double(raw.tail_duration_s),
            tailSamples: raw.tail_samples,
            mountTailDriftDeg: Double(raw.mount_tail_drift_deg),
            mountTailStdDeg: Double(raw.mount_tail_std_deg),
            gyroBiasTailDriftRadps: Double(raw.gyro_bias_tail_drift_radps),
            gyroBiasTailStdRadps: Double(raw.gyro_bias_tail_std_radps),
            accelBiasTailDriftMps2: Double(raw.accel_bias_tail_drift_mps2),
            accelBiasTailStdMps2: Double(raw.accel_bias_tail_std_mps2),
            mountSigmaMaxDeg: Double(raw.mount_sigma_max_deg),
            attitudeSigmaMaxDeg: Double(raw.attitude_sigma_max_deg)
        )
    }

    func withContext(
        initialized: Bool,
        mountReady: Bool,
        gnssAccuracy: GnssAccuracy
    ) -> FusionHealth {
        let quality = gnssAccuracy.quality()
        return FusionHealth(
            state: state,
            initialized: initialized,
            mountReady: mountReady,
            gnssQuality: quality,
            fusedConfidence: Self.confidence(
                state: state,
                gnssQuality: quality,
                mountReady: mountReady,
                initialized: initialized
            ),
            running: running,
            stable: stable,
            degraded: degraded,
            navigationUsable: navigationUsable,
            reasonMask: reasonMask,
            postInitTimeSec: postInitTimeSec,
            distanceM: distanceM,
            tailDurationSec: tailDurationSec,
            tailSamples: tailSamples,
            mountTailDriftDeg: mountTailDriftDeg,
            mountTailStdDeg: mountTailStdDeg,
            gyroBiasTailDriftRadps: gyroBiasTailDriftRadps,
            gyroBiasTailStdRadps: gyroBiasTailStdRadps,
            accelBiasTailDriftMps2: accelBiasTailDriftMps2,
            accelBiasTailStdMps2: accelBiasTailStdMps2,
            mountSigmaMaxDeg: mountSigmaMaxDeg,
            attitudeSigmaMaxDeg: attitudeSigmaMaxDeg
        )
    }

    static func evaluate(
        mountReady: Bool,
        initialized: Bool,
        gnssAccuracy: GnssAccuracy,
        streamHealth: StreamHealth? = nil,
        goodHorizontalAccuracyM: Double = 8.0,
        fairHorizontalAccuracyM: Double = 25.0
    ) -> FusionHealth {
        let quality = gnssAccuracy.quality(
            goodHorizontalM: goodHorizontalAccuracyM,
            fairHorizontalM: fairHorizontalAccuracyM
        )

        let state: State
        if let streamHealth, streamHealth.imu.blocksFusion {
            state = initialized ? .degraded : .notReady
        } else if let streamHealth, streamHealth.gnss.blocksFusion {
            state = initialized ? .degraded : .initializing
        } else {
            switch (mountReady, initialized, quality) {
        case (_, _, .unavailable), (_, _, .poor):
            state = initialized ? .degraded : .initializing
        case (false, false, _):
            state = .initializing
        case (false, true, _):
            state = .initializing
        case (true, false, _):
            state = .initializing
        case (true, true, _):
            state = .running
            }
        }

        return FusionHealth(
            state: state,
            initialized: initialized,
            mountReady: mountReady,
            gnssQuality: quality,
            fusedConfidence: confidence(
                state: state,
                gnssQuality: quality,
                mountReady: mountReady,
                initialized: initialized
            )
        )
    }

    private static func confidence(
        state: State,
        gnssQuality: GnssAccuracy.Quality,
        mountReady: Bool,
        initialized: Bool
    ) -> Double {
        if state != .running && state != .stable {
            switch state {
            case .degraded, .degradedDeadReckoning:
                return initialized ? 0.35 : 0.15
            case .initializing:
                return 0.20
            case .notReady, .awaitingGnssReseed:
                return 0.0
            case .running, .stable:
                return 1.0
            }
        }

        switch gnssQuality {
        case .good:
            return state == .stable ? 1.0 : 0.85
        case .fair:
            return state == .stable ? 0.90 : 0.75
        case .poor, .unavailable:
            return mountReady && initialized ? 0.35 : 0.15
        }
    }
}
