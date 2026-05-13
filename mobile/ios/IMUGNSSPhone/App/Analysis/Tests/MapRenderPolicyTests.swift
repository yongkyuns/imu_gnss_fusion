import CoreLocation
import MapKit
import XCTest
@testable import IMUGNSSPhone

@MainActor
final class MapRenderPolicyTests: XCTestCase {
    func testFusedMarkerIsUpdatedInPlaceAcrossHighRateCoordinateUpdates() {
        let mapView = MKMapView()
        let layer = MapAnnotationLayer()

        let firstAction = layer.update(
            on: mapView,
            coordinate: CLLocationCoordinate2D(latitude: 37.0, longitude: -122.0),
            title: "Fused"
        )
        let firstAnnotation = layer.annotation

        XCTAssertEqual(firstAction, .create)
        XCTAssertNotNil(firstAnnotation)
        XCTAssertEqual(mapView.annotations.count, 1)

        for index in 1...100 {
            let action = layer.update(
                on: mapView,
                coordinate: CLLocationCoordinate2D(
                    latitude: 37.0 + Double(index) * 0.000001,
                    longitude: -122.0
                ),
                title: "Fused"
            )
            XCTAssertEqual(action, .move)
        }

        XCTAssertTrue(firstAnnotation === layer.annotation)
        XCTAssertEqual(mapView.annotations.count, 1)
        XCTAssertEqual(layer.annotation?.coordinate.latitude ?? .nan, 37.0001, accuracy: 1e-9)
        XCTAssertEqual(layer.annotation?.coordinate.longitude ?? .nan, -122.0, accuracy: 1e-9)
    }

    func testFusedMarkerIsRemovedOnlyWhenCoordinateDisappears() {
        let mapView = MKMapView()
        let layer = MapAnnotationLayer()

        _ = layer.update(
            on: mapView,
            coordinate: CLLocationCoordinate2D(latitude: 37.0, longitude: -122.0),
            title: "Fused"
        )
        let removeAction = layer.update(on: mapView, coordinate: nil, title: "Fused")
        let repeatedNilAction = layer.update(on: mapView, coordinate: nil, title: "Fused")

        XCTAssertEqual(removeAction, .remove)
        XCTAssertEqual(repeatedNilAction, .none)
        XCTAssertNil(layer.annotation)
        XCTAssertEqual(mapView.annotations.count, 0)
    }

    func testCameraRefitPolicyThrottlesContinuousCoordinateChanges() {
        XCTAssertTrue(
            MapCameraPolicy.shouldRefit(
                previousKey: nil,
                nextKey: "initial",
                elapsedSinceLastRefitSec: nil
            )
        )
        XCTAssertFalse(
            MapCameraPolicy.shouldRefit(
                previousKey: "old",
                nextKey: "new",
                elapsedSinceLastRefitSec: 0.25
            )
        )
        XCTAssertTrue(
            MapCameraPolicy.shouldRefit(
                previousKey: "old",
                nextKey: "new",
                elapsedSinceLastRefitSec: MapCameraPolicy.minimumRefitIntervalSec
            )
        )
        XCTAssertFalse(
            MapCameraPolicy.shouldRefit(
                previousKey: "same",
                nextKey: "same",
                elapsedSinceLastRefitSec: 10.0
            )
        )
    }

    func testFusedMapOutputIsHiddenUntilInitializedAndMountReady() {
        XCTAssertFalse(FusedMapVisibilityPolicy.shouldShowFusedOutput(initialized: false, mountReady: false))
        XCTAssertFalse(FusedMapVisibilityPolicy.shouldShowFusedOutput(initialized: true, mountReady: false))
        XCTAssertFalse(FusedMapVisibilityPolicy.shouldShowFusedOutput(initialized: false, mountReady: true))
        XCTAssertTrue(FusedMapVisibilityPolicy.shouldShowFusedOutput(initialized: true, mountReady: true))
    }
}
