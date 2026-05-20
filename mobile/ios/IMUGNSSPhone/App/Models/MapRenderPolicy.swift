import Foundation
import MapKit

struct MapCoordinateSample: Equatable {
    let latitude: Double
    let longitude: Double

    init?(_ coordinate: CLLocationCoordinate2D?) {
        guard let coordinate, CLLocationCoordinate2DIsValid(coordinate) else { return nil }
        latitude = coordinate.latitude
        longitude = coordinate.longitude
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
}

enum MapRouteOverlayPolicy {
    static let minimumUpdateIntervalSec: TimeInterval = 0.20

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
    private(set) var gnssRouteOverlay: MKPolyline?
    private(set) var fusedRouteOverlay: MKPolyline?

    func update(
        on mapView: MKMapView,
        gnssCoordinates: [CLLocationCoordinate2D],
        fusedCoordinates: [CLLocationCoordinate2D]
    ) {
        let previousGnssOverlay = gnssRouteOverlay
        let previousFusedOverlay = fusedRouteOverlay
        let nextGnssOverlay = Self.routeOverlay(coordinates: gnssCoordinates, title: "gnss")
        let nextFusedOverlay = Self.routeOverlay(coordinates: fusedCoordinates, title: "fused")

        if let nextGnssOverlay {
            gnssRouteOverlay = nextGnssOverlay
            if let previousFusedOverlay {
                mapView.insertOverlay(nextGnssOverlay, below: previousFusedOverlay)
            } else {
                mapView.addOverlay(nextGnssOverlay, level: .aboveRoads)
            }
        } else {
            gnssRouteOverlay = nil
        }

        if let previousGnssOverlay {
            mapView.removeOverlay(previousGnssOverlay)
        }

        if let nextFusedOverlay {
            fusedRouteOverlay = nextFusedOverlay
            if let nextGnssOverlay {
                mapView.insertOverlay(nextFusedOverlay, above: nextGnssOverlay)
            } else if let previousFusedOverlay {
                mapView.insertOverlay(nextFusedOverlay, above: previousFusedOverlay)
            } else {
                mapView.addOverlay(nextFusedOverlay, level: .aboveRoads)
            }
        } else {
            fusedRouteOverlay = nil
        }

        if let previousFusedOverlay {
            mapView.removeOverlay(previousFusedOverlay)
        }
    }

    private static func routeOverlay(
        coordinates: [CLLocationCoordinate2D],
        title: String
    ) -> MKPolyline? {
        guard coordinates.count >= 2 else { return nil }
        let polyline = MKPolyline(coordinates: coordinates, count: coordinates.count)
        polyline.title = title
        return polyline
    }
}

enum FusedMapVisibilityPolicy {
    static func shouldShowFusedOutput(initialized: Bool, mountReady: Bool) -> Bool {
        initialized && mountReady
    }
}
