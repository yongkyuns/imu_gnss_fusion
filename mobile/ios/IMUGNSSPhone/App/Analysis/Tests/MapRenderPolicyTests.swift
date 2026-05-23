import CoreLocation
import MapKit
import UIKit
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

    func testFollowPolicySuspendsCameraUpdatesDuringUserGestures() {
        let now = Date(timeIntervalSince1970: 100)

        XCTAssertFalse(
            MapFollowPolicy.shouldApplyFollowCamera(
                followsCurrentMarker: false,
                isUserInteracting: false,
                suspendedUntil: nil,
                now: now
            )
        )
        XCTAssertFalse(
            MapFollowPolicy.shouldApplyFollowCamera(
                followsCurrentMarker: true,
                isUserInteracting: true,
                suspendedUntil: nil,
                now: now
            )
        )
        XCTAssertFalse(
            MapFollowPolicy.shouldApplyFollowCamera(
                followsCurrentMarker: true,
                isUserInteracting: false,
                suspendedUntil: now.addingTimeInterval(0.5),
                now: now
            )
        )
        XCTAssertTrue(
            MapFollowPolicy.shouldApplyFollowCamera(
                followsCurrentMarker: true,
                isUserInteracting: false,
                suspendedUntil: now.addingTimeInterval(-0.1),
                now: now
            )
        )
        XCTAssertGreaterThan(
            MapFollowPolicy.followSuspendedUntil(afterUserInteractionAt: now),
            now
        )
    }

    func testDisplayUnitPolicyFormatsUserFacingSpeedsInKmh() {
        XCTAssertEqual(
            DisplayUnitPolicy.speedKmhText(fromMetersPerSecond: 10.0, decimals: 1),
            "36.0"
        )
        XCTAssertEqual(
            DisplayUnitPolicy.velocityKmhText(fromMetersPerSecond: -2.5, decimals: 1),
            "-9.0"
        )
        XCTAssertEqual(
            DisplayUnitPolicy.speedKmhText(fromMetersPerSecond: nil, decimals: 1),
            "-"
        )
        XCTAssertEqual(
            DisplayUnitPolicy.speedKmhText(fromMetersPerSecond: .nan, decimals: 1),
            "-"
        )
    }

    func testMotionEventHeadsUpPolicyOnlyPresentsNewEvents() {
        XCTAssertEqual(MotionEventHeadsUpPolicy.displayDurationSec, 4.0, accuracy: 1e-12)
        XCTAssertFalse(MotionEventHeadsUpPolicy.shouldPresent(latestID: nil, displayedID: nil))
        XCTAssertTrue(MotionEventHeadsUpPolicy.shouldPresent(latestID: "event-1", displayedID: nil))
        XCTAssertFalse(MotionEventHeadsUpPolicy.shouldPresent(latestID: "event-1", displayedID: "event-1"))
        XCTAssertTrue(MotionEventHeadsUpPolicy.shouldPresent(latestID: "event-2", displayedID: "event-1"))
    }

    func testMotionEventVisualPolicyUsesRecognizableAvailableSymbols() {
        XCTAssertEqual(MotionEventVisualPolicy.systemImage(for: .reverse), "arrow.backward")
        XCTAssertEqual(MotionEventVisualPolicy.systemImage(for: .speedBump), "road.lanes")
        XCTAssertEqual(MotionEventVisualPolicy.uiColor(for: .speedBump), .systemOrange)
        XCTAssertEqual(MotionEventVisualPolicy.uiColor(for: .uphill), .systemMint)

        for kind in MotionEvent.Kind.allCases {
            XCTAssertNotNil(
                UIImage(systemName: MotionEventVisualPolicy.systemImage(for: kind)),
                "Missing SF Symbol for \(kind.rawValue)"
            )
        }
    }

    func testGeographicRoutePolicyUsesSnapshotCoordinatesDirectly() {
        let coordinates = MapGeographicRoutePolicy.coordinates(
            history: [
                GeographicCoordinate(latitudeDeg: 37.0, longitudeDeg: -122.0),
                GeographicCoordinate(latitudeDeg: 37.0001, longitudeDeg: -122.0002)
            ],
            current: GeographicCoordinate(latitudeDeg: 37.0002, longitudeDeg: -122.0003)
        )

        XCTAssertEqual(coordinates.count, 3)
        XCTAssertEqual(coordinates[0].latitude, 37.0, accuracy: 1.0e-12)
        XCTAssertEqual(coordinates[0].longitude, -122.0, accuracy: 1.0e-12)
        XCTAssertEqual(coordinates[1].latitude, 37.0001, accuracy: 1.0e-12)
        XCTAssertEqual(coordinates[1].longitude, -122.0002, accuracy: 1.0e-12)
        XCTAssertEqual(coordinates[2].latitude, 37.0002, accuracy: 1.0e-12)
        XCTAssertEqual(coordinates[2].longitude, -122.0003, accuracy: 1.0e-12)
    }

    func testGeographicRoutePolicyFiltersInvalidCoordinatesAndAvoidsDuplicateCurrentPoint() {
        let coordinates = MapGeographicRoutePolicy.coordinates(
            history: [
                GeographicCoordinate(latitudeDeg: 120.0, longitudeDeg: -122.0),
                GeographicCoordinate(latitudeDeg: 37.0, longitudeDeg: -122.0)
            ],
            current: GeographicCoordinate(latitudeDeg: 37.0, longitudeDeg: -122.0)
        )

        XCTAssertEqual(coordinates.count, 1)
        XCTAssertEqual(coordinates[0].latitude, 37.0, accuracy: 1.0e-12)
        XCTAssertEqual(coordinates[0].longitude, -122.0, accuracy: 1.0e-12)
    }

    func testAlignProgressPolicyUsesTiltThenYawCovarianceCollapse() {
        XCTAssertEqual(
            AlignProgressPolicy.progress(.unavailable, mountReady: false),
            0.0,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            AlignProgressPolicy.progress(.unavailable, mountReady: true),
            1.0,
            accuracy: 1e-12
        )

        let initial = AlignProgressSnapshot(
            isValid: true,
            rollSigmaDeg: 10.0,
            pitchSigmaDeg: 10.0,
            yawSigmaDeg: 60.0
        )
        XCTAssertEqual(AlignProgressPolicy.progress(initial, mountReady: false), 0.0, accuracy: 1e-12)

        let tiltReady = AlignProgressSnapshot(
            isValid: true,
            rollSigmaDeg: 5.0,
            pitchSigmaDeg: 5.0,
            yawSigmaDeg: 60.0
        )
        XCTAssertEqual(AlignProgressPolicy.progress(tiltReady, mountReady: false), 0.3, accuracy: 1e-12)

        let halfway = AlignProgressSnapshot(
            isValid: true,
            rollSigmaDeg: 7.5,
            pitchSigmaDeg: 7.5,
            yawSigmaDeg: 34.0
        )
        XCTAssertEqual(AlignProgressPolicy.progress(halfway, mountReady: false), 0.5, accuracy: 1e-12)

        let ready = AlignProgressSnapshot(
            isValid: true,
            rollSigmaDeg: 5.0,
            pitchSigmaDeg: 5.0,
            yawSigmaDeg: 8.0
        )
        XCTAssertEqual(AlignProgressPolicy.progressPercent(ready, mountReady: false), 100)
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

    func testRouteOverlayLayerKeepsFusedRouteAboveGnssRouteWithoutReplacingStableOverlays() {
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
        XCTAssertTrue(layer.gnssRouteOverlay === previousGnss)
        XCTAssertTrue(layer.fusedRouteOverlay === previousFused)
        XCTAssertTrue(mapView.overlays.first === layer.gnssRouteOverlay)
        XCTAssertTrue(mapView.overlays.last === layer.fusedRouteOverlay)
        XCTAssertEqual(layer.gnssRouteOverlay?.coordinates.count, 3)
        XCTAssertEqual(layer.fusedRouteOverlay?.coordinates.count, 3)
    }

    func testRouteOverlayLayerOnlyRemovesOverlayWhenRouteDisappears() {
        let mapView = MKMapView()
        let layer = MapRouteOverlayLayer()
        let gnss = [
            CLLocationCoordinate2D(latitude: 37.0, longitude: -122.0),
            CLLocationCoordinate2D(latitude: 37.001, longitude: -122.0)
        ]
        let fused = [
            CLLocationCoordinate2D(latitude: 37.0, longitude: -122.0001),
            CLLocationCoordinate2D(latitude: 37.001, longitude: -122.0001)
        ]

        layer.update(on: mapView, gnssCoordinates: gnss, fusedCoordinates: fused)
        let fusedOverlay = layer.fusedRouteOverlay

        layer.update(on: mapView, gnssCoordinates: [], fusedCoordinates: fused)

        XCTAssertNil(layer.gnssRouteOverlay)
        XCTAssertTrue(layer.fusedRouteOverlay === fusedOverlay)
        XCTAssertEqual(mapView.overlays.count, 1)
        XCTAssertTrue(mapView.overlays.first === fusedOverlay)
    }

    func testFusedMapOutputIsVisibleAfterFilterInitialization() {
        XCTAssertFalse(FusedMapVisibilityPolicy.shouldShowFusedOutput(initialized: false, mountReady: false))
        XCTAssertTrue(FusedMapVisibilityPolicy.shouldShowFusedOutput(initialized: true, mountReady: false))
        XCTAssertFalse(FusedMapVisibilityPolicy.shouldShowFusedOutput(initialized: false, mountReady: true))
        XCTAssertTrue(FusedMapVisibilityPolicy.shouldShowFusedOutput(initialized: true, mountReady: true))
    }
}
