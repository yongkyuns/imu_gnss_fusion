import Foundation
import MapKit
import UIKit

struct MapCoordinateSample: Equatable {
    let latitude: Double
    let longitude: Double

    init?(_ coordinate: CLLocationCoordinate2D?) {
        guard let coordinate, CLLocationCoordinate2DIsValid(coordinate) else { return nil }
        latitude = coordinate.latitude
        longitude = coordinate.longitude
    }
}

enum MapGeographicRoutePolicy {
    static func coordinates(
        history: [GeographicCoordinate],
        current: GeographicCoordinate?
    ) -> [CLLocationCoordinate2D] {
        var coordinates = history.compactMap(mapCoordinate)
        if let currentCoordinate = mapCoordinate(current),
           shouldAppendCurrentCoordinate(currentCoordinate, to: coordinates) {
            coordinates.append(currentCoordinate)
        }
        return coordinates
    }

    private static func shouldAppendCurrentCoordinate(
        _ current: CLLocationCoordinate2D,
        to coordinates: [CLLocationCoordinate2D]
    ) -> Bool {
        guard let last = coordinates.last else { return true }
        return abs(last.latitude - current.latitude) >= 1.0e-8
            || abs(last.longitude - current.longitude) >= 1.0e-8
    }

    private static func mapCoordinate(_ coordinate: GeographicCoordinate?) -> CLLocationCoordinate2D? {
        guard let coordinate, coordinate.isValidLatitudeLongitude else { return nil }
        let mapCoordinate = CLLocationCoordinate2D(
            latitude: coordinate.latitudeDeg,
            longitude: coordinate.longitudeDeg
        )
        return CLLocationCoordinate2DIsValid(mapCoordinate) ? mapCoordinate : nil
    }
}

enum MapAnnotationAction: Equatable {
    case none
    case create
    case move
    case remove
}

enum MapAnnotationReconciler {
    static func action(
        hasAnnotation: Bool,
        current: MapCoordinateSample?,
        next: MapCoordinateSample?
    ) -> MapAnnotationAction {
        guard let next else {
            return hasAnnotation ? .remove : .none
        }
        guard hasAnnotation else {
            return .create
        }
        return current == next ? .none : .move
    }
}

@MainActor
final class MapAnnotationLayer {
    private(set) var annotation: MKPointAnnotation?
    private var currentCoordinate: MapCoordinateSample?

    @discardableResult
    func update(
        on mapView: MKMapView,
        coordinate: CLLocationCoordinate2D?,
        title: String
    ) -> MapAnnotationAction {
        let nextCoordinate = MapCoordinateSample(coordinate)
        let action = MapAnnotationReconciler.action(
            hasAnnotation: annotation != nil,
            current: currentCoordinate,
            next: nextCoordinate
        )

        switch action {
        case .none:
            break
        case .create:
            guard let coordinate else { break }
            let marker = MKPointAnnotation()
            marker.coordinate = coordinate
            marker.title = title
            annotation = marker
            currentCoordinate = nextCoordinate
            mapView.addAnnotation(marker)
        case .move:
            guard let coordinate else { break }
            annotation?.coordinate = coordinate
            currentCoordinate = nextCoordinate
        case .remove:
            if let annotation {
                mapView.removeAnnotation(annotation)
            }
            annotation = nil
            currentCoordinate = nil
        }

        return action
    }
}

enum MapCameraPolicy {
    static func shouldRefit(
        isForced: Bool,
        hasExistingViewport: Bool,
        hasVisibleRoute: Bool
    ) -> Bool {
        isForced || (!hasExistingViewport && hasVisibleRoute)
    }
}

enum MapFollowPolicy {
    static let userInteractionResumeDelaySec: TimeInterval = 1.25

    static func normalizedHeadingDeg(_ headingDeg: Double?) -> Double? {
        guard let headingDeg, headingDeg.isFinite else { return nil }
        var normalized = headingDeg.truncatingRemainder(dividingBy: 360.0)
        if normalized < 0.0 {
            normalized += 360.0
        }
        return normalized
    }

    static func targetCoordinate(
        fusedCoordinate: CLLocationCoordinate2D?,
        gnssCoordinate: CLLocationCoordinate2D?
    ) -> CLLocationCoordinate2D? {
        if let fusedCoordinate, CLLocationCoordinate2DIsValid(fusedCoordinate) {
            return fusedCoordinate
        }
        if let gnssCoordinate, CLLocationCoordinate2DIsValid(gnssCoordinate) {
            return gnssCoordinate
        }
        return nil
    }

    static func shouldApplyFollowCamera(
        followsCurrentMarker: Bool,
        isUserInteracting: Bool,
        suspendedUntil: Date?,
        now: Date
    ) -> Bool {
        guard followsCurrentMarker else { return false }
        if isUserInteracting { return false }
        if let suspendedUntil, now < suspendedUntil { return false }
        return true
    }

    static func followSuspendedUntil(afterUserInteractionAt now: Date) -> Date {
        now.addingTimeInterval(userInteractionResumeDelaySec)
    }
}

enum DisplayUnitPolicy {
    static func kilometersPerHour(fromMetersPerSecond valueMps: Double?) -> Double? {
        guard let valueMps, valueMps.isFinite else { return nil }
        return valueMps * 3.6
    }

    static func speedKmhText(fromMetersPerSecond valueMps: Double?, decimals: Int) -> String {
        guard let valueKmh = kilometersPerHour(fromMetersPerSecond: valueMps) else { return "-" }
        return String(format: "%.*f", decimals, valueKmh)
    }

    static func velocityKmhText(fromMetersPerSecond valueMps: Double?, decimals: Int) -> String {
        speedKmhText(fromMetersPerSecond: valueMps, decimals: decimals)
    }
}

enum MotionEventHeadsUpPolicy {
    static let displayDurationSec: TimeInterval = 4.0

    static func shouldPresent(latestID: String?, displayedID: String?) -> Bool {
        guard let latestID else { return false }
        return latestID != displayedID
    }
}

enum MotionEventVisualPolicy {
    static func systemImage(for kind: MotionEvent.Kind) -> String {
        switch kind {
        case .reverse: return "arrow.backward"
        case .harshAcceleration: return "bolt.fill"
        case .harshBraking: return "exclamationmark.octagon.fill"
        case .harshCornering: return "arrow.turn.up.right"
        case .speedBump: return "road.lanes"
        case .roadShock: return "waveform.path.ecg"
        case .roughRoad: return "point.3.connected.trianglepath.dotted"
        case .downhill: return "arrow.down.right"
        case .uphill: return "arrow.up.right"
        case .gnssDegraded: return "location.slash.fill"
        case .mountReady: return "mappin.and.ellipse"
        case .fusionReady: return "checkmark.seal.fill"
        }
    }

    static func uiColor(for kind: MotionEvent.Kind) -> UIColor {
        switch kind {
        case .reverse: return .systemPurple
        case .harshAcceleration: return .systemGreen
        case .harshBraking: return .systemRed
        case .harshCornering: return .systemPink
        case .speedBump: return .systemOrange
        case .roadShock: return .systemOrange
        case .roughRoad: return .systemBrown
        case .downhill: return .systemIndigo
        case .uphill: return .systemMint
        case .gnssDegraded: return .systemOrange
        case .mountReady: return .systemBlue
        case .fusionReady: return .systemTeal
        }
    }
}

enum AlignProgressPolicy {
    static let tiltProgressWeight = 0.30
    static let yawProgressWeight = 0.70
    static let rollReadySigmaDeg = 5.0
    static let pitchReadySigmaDeg = 5.0
    static let yawReadySigmaDeg = 8.0
    static let rollInitialSigmaDeg = 10.0
    static let pitchInitialSigmaDeg = 10.0
    static let yawInitialSigmaDeg = 60.0

    static func progress(_ snapshot: AlignProgressSnapshot, mountReady: Bool) -> Double {
        if mountReady { return 1.0 }
        guard snapshot.isValid else { return 0.0 }
        let tiltProgress = min(
            axisProgress(
                sigmaDeg: snapshot.rollSigmaDeg,
                initialSigmaDeg: rollInitialSigmaDeg,
                readySigmaDeg: rollReadySigmaDeg
            ),
            axisProgress(
                sigmaDeg: snapshot.pitchSigmaDeg,
                initialSigmaDeg: pitchInitialSigmaDeg,
                readySigmaDeg: pitchReadySigmaDeg
            )
        )
        let yawProgress = axisProgress(
            sigmaDeg: snapshot.yawSigmaDeg,
            initialSigmaDeg: yawInitialSigmaDeg,
            readySigmaDeg: yawReadySigmaDeg
        )
        return min(max(tiltProgressWeight * tiltProgress + yawProgressWeight * yawProgress, 0.0), 1.0)
    }

    static func progressPercent(_ snapshot: AlignProgressSnapshot, mountReady: Bool) -> Int {
        Int((progress(snapshot, mountReady: mountReady) * 100.0).rounded())
    }

    static func axisReady(sigmaDeg: Double?, readySigmaDeg: Double) -> Bool {
        guard let sigmaDeg, sigmaDeg.isFinite else { return false }
        return sigmaDeg <= readySigmaDeg
    }

    private static func axisProgress(
        sigmaDeg: Double?,
        initialSigmaDeg: Double,
        readySigmaDeg: Double
    ) -> Double {
        guard let sigmaDeg, sigmaDeg.isFinite else { return 0.0 }
        guard initialSigmaDeg > readySigmaDeg else { return sigmaDeg <= readySigmaDeg ? 1.0 : 0.0 }
        if sigmaDeg <= readySigmaDeg { return 1.0 }
        if sigmaDeg >= initialSigmaDeg { return 0.0 }
        return (initialSigmaDeg - sigmaDeg) / (initialSigmaDeg - readySigmaDeg)
    }
}

enum MapRouteOverlayPolicy {
    static let minimumUpdateIntervalSec: TimeInterval = 0.50

    static func shouldUpdate(
        previousGnssCount: Int?,
        previousFusedCount: Int?,
        nextGnssCount: Int,
        nextFusedCount: Int,
        elapsedSinceLastUpdateSec: TimeInterval?
    ) -> Bool {
        guard let previousGnssCount, let previousFusedCount else { return true }

        let previousVisibility = (previousGnssCount >= 2, previousFusedCount >= 2)
        let nextVisibility = (nextGnssCount >= 2, nextFusedCount >= 2)
        if previousVisibility != nextVisibility {
            return true
        }

        guard let elapsedSinceLastUpdateSec else { return true }
        return elapsedSinceLastUpdateSec >= minimumUpdateIntervalSec
    }
}

@MainActor
final class MapRouteOverlayLayer {
    private(set) var gnssRouteOverlay: MapRouteOverlay?
    private(set) var fusedRouteOverlay: MapRouteOverlay?

    func update(
        on mapView: MKMapView,
        gnssCoordinates: [CLLocationCoordinate2D],
        fusedCoordinates: [CLLocationCoordinate2D]
    ) {
        updateGnssOverlay(on: mapView, coordinates: gnssCoordinates)
        updateFusedOverlay(on: mapView, coordinates: fusedCoordinates)
    }

    private func updateGnssOverlay(on mapView: MKMapView, coordinates: [CLLocationCoordinate2D]) {
        guard coordinates.count >= 2 else {
            removeOverlay(&gnssRouteOverlay, from: mapView)
            return
        }

        if let gnssRouteOverlay {
            gnssRouteOverlay.update(coordinates: coordinates)
            refreshRenderer(for: gnssRouteOverlay, on: mapView)
            return
        }

        let overlay = MapRouteOverlay(kind: .gnss, coordinates: coordinates)
        gnssRouteOverlay = overlay
        if let fusedRouteOverlay {
            mapView.insertOverlay(overlay, below: fusedRouteOverlay)
        } else {
            mapView.addOverlay(overlay, level: .aboveRoads)
        }
    }

    private func updateFusedOverlay(on mapView: MKMapView, coordinates: [CLLocationCoordinate2D]) {
        guard coordinates.count >= 2 else {
            removeOverlay(&fusedRouteOverlay, from: mapView)
            return
        }

        if let fusedRouteOverlay {
            fusedRouteOverlay.update(coordinates: coordinates)
            refreshRenderer(for: fusedRouteOverlay, on: mapView)
            return
        }

        let overlay = MapRouteOverlay(kind: .fused, coordinates: coordinates)
        fusedRouteOverlay = overlay
        if let gnssRouteOverlay {
            mapView.insertOverlay(overlay, above: gnssRouteOverlay)
        } else {
            mapView.addOverlay(overlay, level: .aboveRoads)
        }
    }

    private func removeOverlay(_ overlay: inout MapRouteOverlay?, from mapView: MKMapView) {
        if let overlay {
            mapView.removeOverlay(overlay)
        }
        overlay = nil
    }

    private func refreshRenderer(for overlay: MapRouteOverlay, on mapView: MKMapView) {
        guard let renderer = mapView.renderer(for: overlay) as? MapRouteOverlayRenderer else { return }
        renderer.refreshPath()
    }
}

final class MapRouteOverlay: NSObject, MKOverlay {
    enum Kind {
        case gnss
        case fused
    }

    let kind: Kind
    private(set) var coordinates: [CLLocationCoordinate2D]

    init(kind: Kind, coordinates: [CLLocationCoordinate2D]) {
        self.kind = kind
        self.coordinates = coordinates
        super.init()
    }

    var coordinate: CLLocationCoordinate2D {
        coordinates.last ?? CLLocationCoordinate2D(latitude: 0.0, longitude: 0.0)
    }

    var boundingMapRect: MKMapRect {
        MKMapRect.world
    }

    func update(coordinates: [CLLocationCoordinate2D]) {
        self.coordinates = coordinates
    }
}

final class MapRouteOverlayRenderer: MKOverlayPathRenderer {
    private let routeOverlay: MapRouteOverlay

    init(routeOverlay: MapRouteOverlay) {
        self.routeOverlay = routeOverlay
        super.init(overlay: routeOverlay)
        switch routeOverlay.kind {
        case .fused:
            strokeColor = UIColor.systemBlue.withAlphaComponent(0.90)
            lineWidth = 5
        case .gnss:
            strokeColor = UIColor.systemOrange.withAlphaComponent(0.82)
            lineWidth = 3
        }
        lineCap = .round
        lineJoin = .round
    }

    override func createPath() {
        let path = CGMutablePath()
        guard let firstCoordinate = routeOverlay.coordinates.first else {
            self.path = path
            return
        }

        path.move(to: point(for: MKMapPoint(firstCoordinate)))
        for coordinate in routeOverlay.coordinates.dropFirst() {
            path.addLine(to: point(for: MKMapPoint(coordinate)))
        }
        self.path = path
    }

    func refreshPath() {
        invalidatePath()
        setNeedsDisplay()
    }
}

enum FusedMapVisibilityPolicy {
    static func shouldShowFusedOutput(initialized: Bool, mountReady: Bool) -> Bool {
        initialized
    }
}
