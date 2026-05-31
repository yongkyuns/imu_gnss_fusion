import XCTest
@testable import IMUGNSSPhone

final class MotionEventDetectorTests: XCTestCase {
    func testSystemEventsEmitOnReadinessAndGnssTransitions() {
        var detector = MotionEventDetector()
        let aligning = FusionHealth.evaluate(
            mountReady: false,
            initialized: false,
            gnssAccuracy: GnssAccuracy(horizontalAccuracyM: 4.0)
        )
        let ready = readyHealth()
        let poor = FusionHealth.evaluate(
            mountReady: true,
            initialized: true,
            gnssAccuracy: GnssAccuracy(horizontalAccuracyM: 80.0)
        )

        _ = detector.updateSystemEvents(sample(tSec: 0.0, health: aligning, initialized: false, mountReady: false))
        let readyEvents = detector.updateSystemEvents(sample(tSec: 0.1, health: ready, initialized: true, mountReady: true))
        let poorEvents = detector.updateSystemEvents(sample(tSec: 0.2, health: poor, initialized: true, mountReady: true))

        XCTAssertTrue(readyEvents.contains { $0.kind == .fusionReady })
        XCTAssertTrue(readyEvents.contains { $0.kind == .mountReady })
        XCTAssertTrue(poorEvents.contains { $0.kind == .gnssDegraded })
    }

    func testGnssDegradedUsesBooleanEdgeNotStateName() {
        var detector = MotionEventDetector()
        let runningButDegraded = FusionHealth(
            state: .running,
            initialized: true,
            mountReady: true,
            gnssQuality: .good,
            fusedConfidence: 0.8,
            degraded: true,
            navigationUsable: true,
            reasonMask: 1 << 9
        )

        let first = detector.updateSystemEvents(
            sample(tSec: 1.0, health: runningButDegraded, initialized: true, mountReady: true)
        )
        let second = detector.updateSystemEvents(
            sample(tSec: 1.1, health: runningButDegraded, initialized: true, mountReady: true)
        )

        XCTAssertEqual(first.filter { $0.kind == .gnssDegraded }.count, 1)
        XCTAssertFalse(second.contains { $0.kind == .gnssDegraded })
    }

    func testRoadEventMappingUsesRustFfiEventKind() {
        let event = MotionEvent(
            roadEvent: RoadEventDetection(
                kind: .harshBraking,
                tSec: 8.0,
                startTSec: 7.2,
                endTSec: 8.0,
                durationSec: 0.8,
                value: 3.5,
                confidence: 0.9
            ),
            sampleTimestamp: Date(timeIntervalSince1970: 10.0),
            currentTSec: 9.0,
            coordinate: nil
        )

        XCTAssertEqual(event.kind, .harshBraking)
        XCTAssertEqual(event.tSec, 8.0, accuracy: 1.0e-12)
        XCTAssertEqual(event.timestamp.timeIntervalSince1970, 9.0, accuracy: 1.0e-12)
        XCTAssertEqual(event.value ?? .nan, 3.5, accuracy: 1.0e-12)
        XCTAssertEqual(event.durationSec ?? .nan, 0.8, accuracy: 1.0e-12)
    }

    private func readyHealth() -> FusionHealth {
        FusionHealth.evaluate(
            mountReady: true,
            initialized: true,
            gnssAccuracy: GnssAccuracy(horizontalAccuracyM: 4.0)
        )
    }

    private func sample(
        tSec: Double,
        health: FusionHealth,
        initialized: Bool,
        mountReady: Bool
    ) -> MotionEventSample {
        MotionEventSample(
            tSec: tSec,
            timestamp: Date(timeIntervalSince1970: tSec),
            coordinate: nil,
            health: health,
            initialized: initialized,
            mountReady: mountReady
        )
    }
}
