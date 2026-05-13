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
    static let minimumRefitIntervalSec = 2.0

    static func shouldRefit(
        previousKey: String?,
        nextKey: String,
        elapsedSinceLastRefitSec: TimeInterval?
    ) -> Bool {
        guard previousKey != nextKey else { return false }
        guard let elapsedSinceLastRefitSec else { return true }
        return elapsedSinceLastRefitSec >= minimumRefitIntervalSec
    }
}

enum FusedMapVisibilityPolicy {
    static func shouldShowFusedOutput(initialized: Bool, mountReady: Bool) -> Bool {
        initialized && mountReady
    }
}
