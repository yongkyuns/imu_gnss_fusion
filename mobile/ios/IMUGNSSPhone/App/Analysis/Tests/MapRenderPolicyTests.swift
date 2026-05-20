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

    func testFollowPolicyPrefersFusedThenFallsBackToGnss() throws {
        let gnss = CLLocationCoordinate2D(latitude: 37.0, longitude: -122.0)
        let fused = CLLocationCoordinate2D(latitude: 37.1, longitude: -122.2)

        var target = try XCTUnwrap(MapFollowPolicy.targetCoordinate(fusedCoordinate: fused, gnssCoordinate: gnss))
        XCTAssertEqual(target.latitude, fused.latitude, accuracy: 1e-12)
        XCTAssertEqual(target.longitude, fused.longitude, accuracy: 1e-12)

        target = try XCTUnwrap(MapFollowPolicy.targetCoordinate(fusedCoordinate: nil, gnssCoordinate: gnss))
        XCTAssertEqual(target.latitude, gnss.latitude, accuracy: 1e-12)
        XCTAssertEqual(target.longitude, gnss.longitude, accuracy: 1e-12)

        XCTAssertNil(
            MapFollowPolicy.targetCoordinate(
                fusedCoordinate: CLLocationCoordinate2D(latitude: .nan, longitude: .nan),
                gnssCoordinate: nil
            )
        )
    }

    func testFollowPolicyNormalizesHeadingForMapCamera() {
        XCTAssertEqual(MapFollowPolicy.normalizedHeadingDeg(0.0) ?? .nan, 0.0, accuracy: 1e-12)
        XCTAssertEqual(MapFollowPolicy.normalizedHeadingDeg(361.0) ?? .nan, 1.0, accuracy: 1e-12)
        XCTAssertEqual(MapFollowPolicy.normalizedHeadingDeg(-1.0) ?? .nan, 359.0, accuracy: 1e-12)
        XCTAssertNil(MapFollowPolicy.normalizedHeadingDeg(.nan))
        XCTAssertNil(MapFollowPolicy.normalizedHeadingDeg(nil))
    }

    func testRouteOverlayPolicyThrottlesContinuousRouteGrowth() {
        XCTAssertLessThanOrEqual(MapRouteOverlayPolicy.minimumUpdateIntervalSec, 0.25)
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

    func testRouteOverlayLayerKeepsFusedRouteAboveGnssRoute() {
        let mapView = MKMapView()
        let layer = MapRouteOverlayLayer()
        let firstGnss = [
            CLLocationCoordinate2D(latitude: 37.0, longitude: -122.0),
            CLLocationCoordinate2D(latitude: 37.001, longitude: -122.0)
        ]
        let firstFused = [
            CLLocationCoordinate2D(latitude: 37.0, longitude: -122.0001),
            CLLocationCoordinate2D(latitude: 37.001, longitude: -122.0001)
        ]

        layer.update(on: mapView, gnssCoordinates: firstGnss, fusedCoordinates: firstFused)

        XCTAssertEqual(mapView.overlays.count, 2)
        XCTAssertTrue(mapView.overlays.first === layer.gnssRouteOverlay)
        XCTAssertTrue(mapView.overlays.last === layer.fusedRouteOverlay)

        let previousGnss = layer.gnssRouteOverlay
        let previousFused = layer.fusedRouteOverlay
        let nextGnss = firstGnss + [CLLocationCoordinate2D(latitude: 37.002, longitude: -122.0)]
        let nextFused = firstFused + [CLLocationCoordinate2D(latitude: 37.002, longitude: -122.0001)]

        layer.update(on: mapView, gnssCoordinates: nextGnss, fusedCoordinates: nextFused)

        XCTAssertEqual(mapView.overlays.count, 2)
        XCTAssertFalse(mapView.overlays.contains { $0 === previousGnss })
        XCTAssertFalse(mapView.overlays.contains { $0 === previousFused })
        XCTAssertTrue(mapView.overlays.first === layer.gnssRouteOverlay)
        XCTAssertTrue(mapView.overlays.last === layer.fusedRouteOverlay)
    }

    func testFusedMapOutputIsHiddenUntilInitializedAndMountReady() {
        XCTAssertFalse(FusedMapVisibilityPolicy.shouldShowFusedOutput(initialized: false, mountReady: false))
        XCTAssertFalse(FusedMapVisibilityPolicy.shouldShowFusedOutput(initialized: true, mountReady: false))
        XCTAssertFalse(FusedMapVisibilityPolicy.shouldShowFusedOutput(initialized: false, mountReady: true))
        XCTAssertTrue(FusedMapVisibilityPolicy.shouldShowFusedOutput(initialized: true, mountReady: true))
    }
}
