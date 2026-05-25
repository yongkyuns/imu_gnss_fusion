import Darwin
import Foundation

struct AppResourceUsageSnapshot: Equatable, Sendable {
    var imuCallbackHz: Double?
    var gnssCallbackHz: Double?
    var fusionEnqueueHz: Double?
    var fusionUiPublishHz: Double?
    var motionUiPublishHz: Double?
    var roadEventUpdateHz: Double?
    var maxFusionQueueDepth: Int = 0
    var droppedFusionOperations: Int = 0
    var residentMemoryMB: Double?

    static let empty = AppResourceUsageSnapshot()
}

struct AppResourceUsageProfiler: Equatable, Sendable {
    private var intervalStartSec: TimeInterval?
    private var imuCallbacks = 0
    private var gnssCallbacks = 0
    private var fusionEnqueues = 0
    private var fusionUiPublishes = 0
    private var motionUiPublishes = 0
    private var roadEventUpdates = 0
    private var intervalMaxFusionQueueDepth = 0
    private var totalDroppedFusionOperations = 0

    mutating func reset(now: TimeInterval? = nil) {
        self = AppResourceUsageProfiler()
        intervalStartSec = now
    }

    mutating func recordImuCallback(now: TimeInterval) {
        ensureIntervalStarted(now: now)
        imuCallbacks += 1
    }

    mutating func recordGnssCallback(now: TimeInterval) {
        ensureIntervalStarted(now: now)
        gnssCallbacks += 1
    }

    mutating func recordFusionEnqueue(queueDepth: Int, now: TimeInterval) {
        ensureIntervalStarted(now: now)
        fusionEnqueues += 1
        intervalMaxFusionQueueDepth = max(intervalMaxFusionQueueDepth, queueDepth)
    }

    mutating func recordFusionDrop(queueDepth: Int, now: TimeInterval) {
        ensureIntervalStarted(now: now)
        totalDroppedFusionOperations += 1
        intervalMaxFusionQueueDepth = max(intervalMaxFusionQueueDepth, queueDepth)
    }

    mutating func recordFusionUiPublish(now: TimeInterval) {
        ensureIntervalStarted(now: now)
        fusionUiPublishes += 1
    }

    mutating func recordMotionUiPublish(now: TimeInterval) {
        ensureIntervalStarted(now: now)
        motionUiPublishes += 1
    }

    mutating func recordRoadEventUpdate(now: TimeInterval) {
        ensureIntervalStarted(now: now)
        roadEventUpdates += 1
    }

    mutating func snapshot(now: TimeInterval, residentMemoryMB: Double?) -> AppResourceUsageSnapshot {
        ensureIntervalStarted(now: now)
        let elapsed = max(now - (intervalStartSec ?? now), 0.001)
        let snapshot = AppResourceUsageSnapshot(
            imuCallbackHz: rate(imuCallbacks, elapsed: elapsed),
            gnssCallbackHz: rate(gnssCallbacks, elapsed: elapsed),
            fusionEnqueueHz: rate(fusionEnqueues, elapsed: elapsed),
            fusionUiPublishHz: rate(fusionUiPublishes, elapsed: elapsed),
            motionUiPublishHz: rate(motionUiPublishes, elapsed: elapsed),
            roadEventUpdateHz: rate(roadEventUpdates, elapsed: elapsed),
            maxFusionQueueDepth: intervalMaxFusionQueueDepth,
            droppedFusionOperations: totalDroppedFusionOperations,
            residentMemoryMB: residentMemoryMB
        )
        reset(now: now)
        totalDroppedFusionOperations = snapshot.droppedFusionOperations
        return snapshot
    }

    private mutating func ensureIntervalStarted(now: TimeInterval) {
        if intervalStartSec == nil {
            intervalStartSec = now
        }
    }

    private func rate(_ count: Int, elapsed: TimeInterval) -> Double? {
        guard count > 0 else { return nil }
        return Double(count) / elapsed
    }
}

enum AppResourceUsageSampler {
    static func residentMemoryMB() -> Double? {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info_data_t>.size / MemoryLayout<natural_t>.size
        )
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return Double(info.resident_size) / (1024.0 * 1024.0)
    }
}
