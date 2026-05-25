import Foundation

enum FusionProfilingLoop: Equatable, Sendable {
    case imu
    case gnss
}

struct FusionLoopProfilingStats: Equatable, Sendable {
    var sampleCount: Int = 0
    var averageMs: Double?
    var lastMs: Double?
}

struct FusionProfilingSnapshot: Equatable, Sendable {
    var imu: FusionLoopProfilingStats = FusionLoopProfilingStats()
    var gnss: FusionLoopProfilingStats = FusionLoopProfilingStats()

    static let empty = FusionProfilingSnapshot()
}

struct FusionLoopProfiler: Equatable, Sendable {
    private var imuTotalSec: Double = 0.0
    private var imuCount: Int = 0
    private var imuLastSec: Double?
    private var gnssTotalSec: Double = 0.0
    private var gnssCount: Int = 0
    private var gnssLastSec: Double?

    var snapshot: FusionProfilingSnapshot {
        FusionProfilingSnapshot(
            imu: stats(totalSec: imuTotalSec, count: imuCount, lastSec: imuLastSec),
            gnss: stats(totalSec: gnssTotalSec, count: gnssCount, lastSec: gnssLastSec)
        )
    }

    mutating func record(loop: FusionProfilingLoop, durationSec: Double) -> FusionProfilingSnapshot {
        guard durationSec.isFinite, durationSec >= 0.0 else { return snapshot }
        switch loop {
        case .imu:
            imuTotalSec += durationSec
            imuCount += 1
            imuLastSec = durationSec
        case .gnss:
            gnssTotalSec += durationSec
            gnssCount += 1
            gnssLastSec = durationSec
        }
        return snapshot
    }

    mutating func reset() {
        self = FusionLoopProfiler()
    }

    private func stats(totalSec: Double, count: Int, lastSec: Double?) -> FusionLoopProfilingStats {
        FusionLoopProfilingStats(
            sampleCount: count,
            averageMs: count > 0 ? (totalSec / Double(count)) * 1_000.0 : nil,
            lastMs: lastSec.map { $0 * 1_000.0 }
        )
    }
}
