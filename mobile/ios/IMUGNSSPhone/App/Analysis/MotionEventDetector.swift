import Foundation

struct MotionEventSample: Equatable, Sendable {
    var tSec: Double
    var timestamp: Date
    var coordinate: GeographicCoordinate?
    var health: FusionHealth
    var initialized: Bool
    var mountReady: Bool
}

struct MotionEventDetector: Sendable {
    private var lastHealthStatus: FusionHealth.Status?
    private var lastInitialized: Bool?
    private var lastMountReady: Bool?

    mutating func reset() {
        self = Self()
    }

    mutating func updateSystemEvents(_ sample: MotionEventSample) -> [MotionEvent] {
        guard sample.tSec.isFinite else { return [] }

        var events: [MotionEvent] = []
        if let lastHealthStatus,
           lastHealthStatus != .poorGNSS,
           sample.health.status == .poorGNSS {
            events.append(systemEvent(sample, kind: .gnssDegraded, confidence: 0.75))
        }
        if let lastInitialized,
           !lastInitialized,
           sample.initialized {
            events.append(systemEvent(sample, kind: .fusionReady))
        }
        if let lastMountReady,
           !lastMountReady,
           sample.mountReady {
            events.append(systemEvent(sample, kind: .mountReady))
        }
        lastHealthStatus = sample.health.status
        lastInitialized = sample.initialized
        lastMountReady = sample.mountReady
        return events
    }

    private func systemEvent(
        _ sample: MotionEventSample,
        kind: MotionEvent.Kind,
        confidence: Double = 1.0
    ) -> MotionEvent {
        MotionEvent(
            kind: kind,
            timestamp: sample.timestamp,
            tSec: sample.tSec,
            coordinate: sample.coordinate,
            confidence: confidence
        )
    }
}
