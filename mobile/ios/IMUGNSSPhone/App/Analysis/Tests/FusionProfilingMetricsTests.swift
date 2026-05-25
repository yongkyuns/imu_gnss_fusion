import XCTest
@testable import IMUGNSSPhone

final class FusionProfilingMetricsTests: XCTestCase {
    func testProfilerTracksImuAndGnssAveragesSeparately() {
        var profiler = FusionLoopProfiler()

        var snapshot = profiler.record(loop: .imu, durationSec: 0.001)
        snapshot = profiler.record(loop: .imu, durationSec: 0.003)
        snapshot = profiler.record(loop: .gnss, durationSec: 0.010)

        XCTAssertEqual(snapshot.imu.sampleCount, 2)
        XCTAssertEqual(snapshot.imu.averageMs ?? .nan, 2.0, accuracy: 1e-12)
        XCTAssertEqual(snapshot.imu.lastMs ?? .nan, 3.0, accuracy: 1e-12)
        XCTAssertEqual(snapshot.gnss.sampleCount, 1)
        XCTAssertEqual(snapshot.gnss.averageMs ?? .nan, 10.0, accuracy: 1e-12)
        XCTAssertEqual(snapshot.gnss.lastMs ?? .nan, 10.0, accuracy: 1e-12)
    }

    func testProfilerIgnoresInvalidDurationsAndResets() {
        var profiler = FusionLoopProfiler()

        _ = profiler.record(loop: .imu, durationSec: .nan)
        XCTAssertEqual(profiler.snapshot, .empty)

        _ = profiler.record(loop: .gnss, durationSec: 0.002)
        profiler.reset()

        XCTAssertEqual(profiler.snapshot, .empty)
    }

    func testResourceProfilerReportsRatesAndPersistentDropCount() {
        var profiler = AppResourceUsageProfiler()

        profiler.recordImuCallback(now: 10.0)
        profiler.recordImuCallback(now: 10.1)
        profiler.recordGnssCallback(now: 10.2)
        profiler.recordFusionEnqueue(queueDepth: 3, now: 10.3)
        profiler.recordFusionDrop(queueDepth: 7, now: 10.4)
        profiler.recordFusionUiPublish(now: 10.5)
        profiler.recordMotionUiPublish(now: 10.6)
        profiler.recordRoadEventUpdate(now: 10.7)

        let snapshot = profiler.snapshot(now: 12.0, residentMemoryMB: 123.4)

        XCTAssertEqual(snapshot.imuCallbackHz ?? .nan, 1.0, accuracy: 1e-12)
        XCTAssertEqual(snapshot.gnssCallbackHz ?? .nan, 0.5, accuracy: 1e-12)
        XCTAssertEqual(snapshot.fusionEnqueueHz ?? .nan, 0.5, accuracy: 1e-12)
        XCTAssertEqual(snapshot.fusionUiPublishHz ?? .nan, 0.5, accuracy: 1e-12)
        XCTAssertEqual(snapshot.motionUiPublishHz ?? .nan, 0.5, accuracy: 1e-12)
        XCTAssertEqual(snapshot.roadEventUpdateHz ?? .nan, 0.5, accuracy: 1e-12)
        XCTAssertEqual(snapshot.maxFusionQueueDepth, 7)
        XCTAssertEqual(snapshot.droppedFusionOperations, 1)
        XCTAssertEqual(snapshot.residentMemoryMB ?? .nan, 123.4, accuracy: 1e-12)

        let nextSnapshot = profiler.snapshot(now: 13.0, residentMemoryMB: nil)
        XCTAssertNil(nextSnapshot.imuCallbackHz)
        XCTAssertEqual(nextSnapshot.droppedFusionOperations, 1)
    }
}
