import XCTest
@testable import IMUGNSSPhone

final class MotionAnalysisTests: XCTestCase {
    func testGroundAnd3DSpeedUseNEDVelocityComponents() {
        let velocity = NavigationVectorNED(north: 3.0, east: 4.0, down: 12.0)

        XCTAssertEqual(MotionKinematics.groundSpeed(velocity), 5.0, accuracy: 1e-12)
        XCTAssertEqual(MotionKinematics.speed3D(velocity), 13.0, accuracy: 1e-12)
    }

    func testIdentityAttitudeMapsNEDVelocityToMatchingFRDVelocity() {
        let velocity = NavigationVectorNED(north: 8.0, east: -2.0, down: 1.5)

        let vehicleVelocity = MotionKinematics.vehicleFRDVelocity(
            qNV: .identity,
            nedVelocityMps: velocity
        )

        XCTAssertEqual(vehicleVelocity.forward, 8.0, accuracy: 1e-12)
        XCTAssertEqual(vehicleVelocity.right, -2.0, accuracy: 1e-12)
        XCTAssertEqual(vehicleVelocity.down, 1.5, accuracy: 1e-12)
    }

    func testEastFacingVehicleMapsEastNEDVelocityToForwardFRDVelocity() {
        let yaw90QNV = Quaternion(
            w: cos(.pi / 4.0),
            x: 0.0,
            y: 0.0,
            z: sin(.pi / 4.0)
        )
        let eastVelocity = NavigationVectorNED(north: 0.0, east: 10.0, down: 0.0)

        let vehicleVelocity = MotionKinematics.vehicleFRDVelocity(
            qNV: yaw90QNV,
            nedVelocityMps: eastVelocity
        )

        XCTAssertEqual(vehicleVelocity.forward, 10.0, accuracy: 1e-12)
        XCTAssertEqual(vehicleVelocity.right, 0.0, accuracy: 1e-12)
        XCTAssertEqual(vehicleVelocity.down, 0.0, accuracy: 1e-12)
    }

    func testVehicleMotionDisplayDerivesCurvatureSideslipAndSegment() {
        let health = FusionHealth.evaluate(
            mountReady: true,
            initialized: true,
            gnssAccuracy: GnssAccuracy(horizontalAccuracyM: 4.0)
        )

        let display = VehicleMotionDisplay.make(
            nedVelocityMps: NavigationVectorNED(north: 10.0, east: 1.0, down: 0.0),
            attitudeQNV: .identity,
            yawRateRadps: 0.2,
            longitudinalAccelerationMps2: 0.0,
            verticalAccelerationMps2: 0.0,
            health: health
        )

        XCTAssertEqual(display.groundSpeedMps, sqrt(101.0), accuracy: 1e-12)
        XCTAssertEqual(display.sideslipRad ?? .nan, atan2(1.0, 10.0), accuracy: 1e-12)
        XCTAssertEqual(display.curvaturePerM ?? .nan, 0.2 / sqrt(101.0), accuracy: 1e-12)
        XCTAssertEqual(display.segment, .turning)
    }

    func testFusionHealthPrioritizesGnssQualityAndReadiness() {
        XCTAssertEqual(
            FusionHealth.evaluate(
                mountReady: true,
                initialized: true,
                gnssAccuracy: GnssAccuracy(horizontalAccuracyM: 60.0)
            ).status,
            .poorGNSS
        )
        XCTAssertEqual(
            FusionHealth.evaluate(
                mountReady: false,
                initialized: false,
                gnssAccuracy: GnssAccuracy(horizontalAccuracyM: 4.0)
            ).status,
            .aligning
        )
        XCTAssertEqual(
            FusionHealth.evaluate(
                mountReady: false,
                initialized: true,
                gnssAccuracy: GnssAccuracy(horizontalAccuracyM: 4.0)
            ).status,
            .alignmentIncomplete
        )
        XCTAssertEqual(
            FusionHealth.evaluate(
                mountReady: true,
                initialized: false,
                gnssAccuracy: GnssAccuracy(horizontalAccuracyM: 4.0)
            ).status,
            .needsMotion
        )
        XCTAssertEqual(
            FusionHealth.evaluate(
                mountReady: true,
                initialized: true,
                gnssAccuracy: GnssAccuracy(horizontalAccuracyM: 4.0)
            ).status,
            .ready
        )
    }

    func testRouteLayerSelectionVisibility() {
        XCTAssertTrue(RouteLayerSelection.delta.showsFusedRoute)
        XCTAssertTrue(RouteLayerSelection.delta.showsGnssRoute)
        XCTAssertTrue(RouteLayerSelection.delta.showsDeltaOverlay)
        XCTAssertFalse(RouteLayerSelection.none.showsFusedRoute)
        XCTAssertFalse(RouteLayerSelection.none.showsGnssRoute)
        XCTAssertFalse(RouteLayerSelection.fused.showsGnssRoute)
        XCTAssertFalse(RouteLayerSelection.gnss.showsFusedRoute)
    }

    func testRouteLayerLegendToggles() {
        XCTAssertEqual(RouteLayerSelection.both.togglingFusedRoute(), .gnss)
        XCTAssertEqual(RouteLayerSelection.both.togglingGnssRoute(), .fused)
        XCTAssertEqual(RouteLayerSelection.gnss.togglingFusedRoute(), .both)
        XCTAssertEqual(RouteLayerSelection.fused.togglingGnssRoute(), .both)
        XCTAssertEqual(RouteLayerSelection.none.togglingFusedRoute(), .fused)
        XCTAssertEqual(RouteLayerSelection.none.togglingGnssRoute(), .gnss)
    }
}
