import Foundation

enum FusionTimingPolicy {
    static let maxGnssImuGapSec: TimeInterval = 1.0
    static let timestampEpsilonSec: TimeInterval = 0.001

    static func hasFreshImu(
        for gnssSampleDate: Date,
        lastImuSampleDate: Date?,
        maxGapSec: TimeInterval = maxGnssImuGapSec
    ) -> Bool {
        guard let lastImuSampleDate else { return false }
        return abs(gnssSampleDate.timeIntervalSince(lastImuSampleDate)) <= maxGapSec
    }

    static func processingDate(
        for sampleDate: Date,
        after lastProcessedDate: Date?,
        epsilonSec: TimeInterval = timestampEpsilonSec
    ) -> Date {
        guard let lastProcessedDate else { return sampleDate }
        guard sampleDate <= lastProcessedDate else { return sampleDate }
        return lastProcessedDate.addingTimeInterval(epsilonSec)
    }
}
