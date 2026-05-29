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
    private var lastHealthState: FusionHealth.State?
    private var lastInitialized: Bool?
    private var lastMountReady: Bool?

    mutating func reset() {
        self = Self()
    }

    mutating func updateSystemEvents(_ sample: MotionEventSample) -> [MotionEvent] {
        guard sample.tSec.isFinite else { return [] }

        var events: [MotionEvent] = []
        let wasDegraded = lastHealthState.map(Self.isDegradedState) ?? false
        if !wasDegraded, sample.health.degraded {
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
        lastHealthState = sample.health.state
        lastInitialized = sample.initialized
        lastMountReady = sample.mountReady
        return events
    }

    private static func isDegradedState(_ state: FusionHealth.State) -> Bool {
        switch state {
        case .degraded, .degradedDeadReckoning, .awaitingGnssReseed:
            return true
        case .notReady, .initializing, .running, .stable:
            return false
        }
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
