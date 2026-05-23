import Foundation

struct MotionEvent: Equatable, Identifiable, Sendable {
    enum Kind: String, CaseIterable, Equatable, Sendable {
        case reverse
        case harshAcceleration
        case harshBraking
        case harshCornering
        case speedBump
        case downhill
        case uphill
        case gnssDegraded
        case mountReady
        case fusionReady

        var displayTitle: String {
            switch self {
            case .reverse: return "Reverse"
            case .harshAcceleration: return "Hard Accel"
            case .harshBraking: return "Hard Brake"
            case .harshCornering: return "Hard Corner"
            case .speedBump: return "Speed Bump"
            case .downhill: return "Downhill"
            case .uphill: return "Uphill"
            case .gnssDegraded: return "GNSS Degraded"
            case .mountReady: return "Mount Ready"
            case .fusionReady: return "Fusion Ready"
            }
        }
    }

    var id: String
    var kind: Kind
    var timestamp: Date
    var tSec: Double
    var coordinate: GeographicCoordinate?
    var value: Double?
    var durationSec: Double?
    var confidence: Double

    init(
        id: String? = nil,
        kind: Kind,
        timestamp: Date,
        tSec: Double,
        coordinate: GeographicCoordinate? = nil,
        value: Double? = nil,
        durationSec: Double? = nil,
        confidence: Double = 1.0
    ) {
        self.kind = kind
        self.timestamp = timestamp
        self.tSec = tSec
        self.coordinate = coordinate?.isValidLatitudeLongitude == true ? coordinate : nil
        self.value = Self.validFinite(value)
        self.durationSec = Self.validNonNegative(durationSec)
        self.confidence = min(max(confidence, 0.0), 1.0)
        self.id = id ?? "\(kind.rawValue)-\(String(format: "%.2f", tSec))"
    }

    init(
        roadEvent: RoadEventDetection,
        sampleTimestamp: Date,
        currentTSec: Double,
        coordinate: GeographicCoordinate?
    ) {
        let kind = Kind(roadEventKind: roadEvent.kind)
        let eventTSec = roadEvent.tSec.isFinite ? roadEvent.tSec : currentTSec
        let timestamp = sampleTimestamp.addingTimeInterval(eventTSec - currentTSec)
        self.init(
            kind: kind,
            timestamp: timestamp,
            tSec: eventTSec,
            coordinate: coordinate,
            value: roadEvent.value,
            durationSec: roadEvent.durationSec,
            confidence: roadEvent.confidence
        )
    }

    private static func validFinite(_ value: Double?) -> Double? {
        guard let value, value.isFinite else { return nil }
        return value
    }

    private static func validNonNegative(_ value: Double?) -> Double? {
        guard let value, value.isFinite, value >= 0.0 else { return nil }
        return value
    }
}

private extension MotionEvent.Kind {
    init(roadEventKind: RoadEventDetection.Kind) {
        switch roadEventKind {
        case .harshAcceleration: self = .harshAcceleration
        case .harshBraking: self = .harshBraking
        case .harshCornering: self = .harshCornering
        case .reverse: self = .reverse
        case .speedBump: self = .speedBump
        case .uphill: self = .uphill
        case .downhill: self = .downhill
        }
    }
}
