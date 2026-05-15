import Foundation

struct StreamHealth: Equatable, Sendable {
    enum ChannelState: Equatable, Sendable {
        case waiting
        case live
        case stale
        case unavailable
        case error(String)
    }

    enum Severity: Equatable, Sendable {
        case nominal
        case warning
        case critical
    }

    var imu: ChannelState
    var gnss: ChannelState
    var recording: ChannelState

    var severity: Severity {
        if imu.isCritical || gnss.isCritical || recording.isCritical {
            return .critical
        }
        if imu != .live || gnss != .live || recording.isWarning {
            return .warning
        }
        return .nominal
    }

    var shortTitle: String {
        switch severity {
        case .nominal:
            return "Streams"
        case .warning:
            if imu == .stale { return "IMU Stale" }
            if gnss == .stale { return "GNSS Stale" }
            return "Check"
        case .critical:
            if imu.isCritical { return "IMU Error" }
            if gnss.isCritical { return "GNSS Error" }
            return "Log Error"
        }
    }

    static let starting = StreamHealth(imu: .waiting, gnss: .waiting, recording: .waiting)

    static func evaluate(
        now: Date,
        lastImuSampleDate: Date?,
        lastGnssSampleDate: Date?,
        motionError: String?,
        locationError: String?,
        recordingError: String?,
        isRecording: Bool,
        imuStaleAfterSec: TimeInterval = 1.0,
        gnssStaleAfterSec: TimeInterval = 5.0
    ) -> StreamHealth {
        StreamHealth(
            imu: channelState(
                now: now,
                lastSampleDate: lastImuSampleDate,
                error: motionError,
                staleAfterSec: imuStaleAfterSec
            ),
            gnss: channelState(
                now: now,
                lastSampleDate: lastGnssSampleDate,
                error: locationError,
                staleAfterSec: gnssStaleAfterSec
            ),
            recording: recordingState(error: recordingError, isRecording: isRecording)
        )
    }

    private static func channelState(
        now: Date,
        lastSampleDate: Date?,
        error: String?,
        staleAfterSec: TimeInterval
    ) -> ChannelState {
        if let error, !error.isEmpty {
            return .error(error)
        }
        guard let lastSampleDate else {
            return .waiting
        }
        return now.timeIntervalSince(lastSampleDate) <= staleAfterSec ? .live : .stale
    }

    private static func recordingState(error: String?, isRecording: Bool) -> ChannelState {
        if let error, !error.isEmpty {
            return .error(error)
        }
        return isRecording ? .live : .waiting
    }
}

private extension StreamHealth.ChannelState {
    var isCritical: Bool {
        switch self {
        case .unavailable, .error:
            return true
        case .waiting, .live, .stale:
            return false
        }
    }

    var isWarning: Bool {
        self == .stale
    }
}
