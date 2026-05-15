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

    func testCameraRefitPolicyOnlyFitsInitiallyOrOnExplicitRecenter() {
        XCTAssertTrue(
            MapCameraPolicy.shouldRefit(
                isForced: false,
                hasExistingViewport: false,
                hasVisibleRoute: true
            )
        )
        XCTAssertFalse(
            MapCameraPolicy.shouldRefit(
                isForced: false,
                hasExistingViewport: true,
                hasVisibleRoute: true
            )
        )
        XCTAssertTrue(
            MapCameraPolicy.shouldRefit(
                isForced: true,
                hasExistingViewport: true,
                hasVisibleRoute: true
            )
        )
        XCTAssertFalse(
            MapCameraPolicy.shouldRefit(
                isForced: false,
                hasExistingViewport: false,
                hasVisibleRoute: false
            )
        )
    }

    func testRouteOverlayPolicyThrottlesContinuousRouteGrowth() {
        XCTAssertTrue(
            MapRouteOverlayPolicy.shouldUpdate(
                previousGnssCount: nil,
                previousFusedCount: nil,
                nextGnssCount: 2,
                nextFusedCount: 0,
                elapsedSinceLastUpdateSec: nil
            )
        )
        XCTAssertFalse(
            MapRouteOverlayPolicy.shouldUpdate(
                previousGnssCount: 100,
                previousFusedCount: 80,
                nextGnssCount: 101,
                nextFusedCount: 81,
                elapsedSinceLastUpdateSec: 0.1
            )
        )
        XCTAssertTrue(
            MapRouteOverlayPolicy.shouldUpdate(
                previousGnssCount: 100,
                previousFusedCount: 80,
                nextGnssCount: 101,
                nextFusedCount: 81,
                elapsedSinceLastUpdateSec: MapRouteOverlayPolicy.minimumUpdateIntervalSec
            )
        )
        XCTAssertTrue(
            MapRouteOverlayPolicy.shouldUpdate(
                previousGnssCount: 100,
                previousFusedCount: 80,
                nextGnssCount: 0,
                nextFusedCount: 80,
                elapsedSinceLastUpdateSec: 0.1
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
