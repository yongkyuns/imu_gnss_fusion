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
    enum Status: String, Equatable, Sendable {
        case ready
        case aligning
        case poorGNSS
        case needsMotion
        case alignmentIncomplete
    }

    var status: Status
    var initialized: Bool
    var mountReady: Bool
    var gnssQuality: GnssAccuracy.Quality
    var fusedConfidence: Double

    init(
        status: Status,
        initialized: Bool,
        mountReady: Bool,
        gnssQuality: GnssAccuracy.Quality,
        fusedConfidence: Double
    ) {
        self.status = status
        self.initialized = initialized
        self.mountReady = mountReady
        self.gnssQuality = gnssQuality
        self.fusedConfidence = min(max(fusedConfidence, 0.0), 1.0)
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

        let status: Status
        if let streamHealth, streamHealth.imu.blocksFusion {
            status = .needsMotion
        } else if let streamHealth, streamHealth.gnss.blocksFusion {
            status = .poorGNSS
        } else {
            switch (mountReady, initialized, quality) {
        case (_, _, .unavailable), (_, _, .poor):
            status = .poorGNSS
        case (false, false, _):
            status = .aligning
        case (false, true, _):
            status = .alignmentIncomplete
        case (true, false, _):
            status = .needsMotion
        case (true, true, _):
            status = .ready
            }
        }

        return FusionHealth(
            status: status,
            initialized: initialized,
            mountReady: mountReady,
            gnssQuality: quality,
            fusedConfidence: confidence(
                status: status,
                gnssQuality: quality,
                mountReady: mountReady,
                initialized: initialized
            )
        )
    }

    private static func confidence(
        status: Status,
        gnssQuality: GnssAccuracy.Quality,
        mountReady: Bool,
        initialized: Bool
    ) -> Double {
        if status != .ready {
            switch status {
            case .poorGNSS:
                return initialized ? 0.35 : 0.15
            case .aligning:
                return 0.20
            case .needsMotion:
                return 0.45
            case .alignmentIncomplete:
                return 0.50
            case .ready:
                return 1.0
            }
        }

        switch gnssQuality {
        case .good:
            return 1.0
        case .fair:
            return 0.75
        case .poor, .unavailable:
            return mountReady && initialized ? 0.35 : 0.15
        }
    }
}
