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
}
