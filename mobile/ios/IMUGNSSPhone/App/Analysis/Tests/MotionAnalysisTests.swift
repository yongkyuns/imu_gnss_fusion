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

    func testVehicleGyroUsesInverseMountRotation() {
        let yaw90QBV = Quaternion(
            w: cos(.pi / 4.0),
            x: 0.0,
            y: 0.0,
            z: sin(.pi / 4.0)
        )

        let vehicleGyro = MotionKinematics.vehicleFRDGyro(
            qBV: yaw90QBV,
            bodyGyroRadps: (x: 0.0, y: 1.0, z: 0.0)
        )

        XCTAssertEqual(vehicleGyro.forward, 1.0, accuracy: 1e-12)
        XCTAssertEqual(vehicleGyro.right, 0.0, accuracy: 1e-12)
        XCTAssertEqual(vehicleGyro.down, 0.0, accuracy: 1e-12)
    }

    func testVehicleAccelerationUsesInverseMountRotation() {
        let yaw90QBV = Quaternion(
            w: cos(.pi / 4.0),
            x: 0.0,
            y: 0.0,
            z: sin(.pi / 4.0)
        )

        let vehicleAccel = MotionKinematics.vehicleFRDAcceleration(
            qBV: yaw90QBV,
            bodyAccelMps2: (x: 0.0, y: 1.0, z: 2.0)
        )

        XCTAssertEqual(vehicleAccel.forward, 1.0, accuracy: 1e-12)
        XCTAssertEqual(vehicleAccel.right, 0.0, accuracy: 1e-12)
        XCTAssertEqual(vehicleAccel.down, 2.0, accuracy: 1e-12)
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
            ).state,
            .degraded
        )
        XCTAssertEqual(
            FusionHealth.evaluate(
                mountReady: false,
                initialized: false,
                gnssAccuracy: GnssAccuracy(horizontalAccuracyM: 4.0)
            ).state,
            .initializing
        )
        XCTAssertEqual(
            FusionHealth.evaluate(
                mountReady: false,
                initialized: true,
                gnssAccuracy: GnssAccuracy(horizontalAccuracyM: 4.0)
            ).state,
            .initializing
        )
        XCTAssertEqual(
            FusionHealth.evaluate(
                mountReady: true,
                initialized: false,
                gnssAccuracy: GnssAccuracy(horizontalAccuracyM: 4.0)
            ).state,
            .initializing
        )
        XCTAssertEqual(
            FusionHealth.evaluate(
                mountReady: true,
                initialized: true,
                gnssAccuracy: GnssAccuracy(horizontalAccuracyM: 4.0)
            ).state,
            .running
        )
    }

    func testFusionHealthMapsAllRustFfiStates() {
        let expected: [(UInt32, FusionHealth.State, Bool, Bool, Bool, Bool)] = [
            (SENSOR_FUSION_STATE_NOT_READY, .notReady, false, false, false, false),
            (SENSOR_FUSION_STATE_INITIALIZING, .initializing, false, false, false, false),
            (SENSOR_FUSION_STATE_RUNNING, .running, true, false, false, true),
            (SENSOR_FUSION_STATE_STABLE, .stable, true, true, false, true),
            (SENSOR_FUSION_STATE_DEGRADED, .degraded, false, false, true, true),
            (SENSOR_FUSION_STATE_DEGRADED_DEAD_RECKONING, .degradedDeadReckoning, false, false, true, true),
            (SENSOR_FUSION_STATE_AWAITING_GNSS_RESEED, .awaitingGnssReseed, false, false, true, false),
        ]

        for (rawState, state, running, stable, degraded, navigationUsable) in expected {
            var raw = SensorFusionFfiHealth()
            raw.state = rawState
            raw.running = running
            raw.stable = stable
            raw.degraded = degraded
            raw.navigation_usable = navigationUsable
            raw.reason_mask = 0xA5
            raw.post_init_time_s = 180.0
            raw.distance_m = 750.0
            raw.tail_samples = 30
            raw.mount_sigma_max_deg = 1.5
            raw.attitude_sigma_max_deg = 4.0

            let health = FusionHealth.from(
                raw: raw,
                initialized: navigationUsable,
                mountReady: true,
                gnssAccuracy: GnssAccuracy(horizontalAccuracyM: 4.0)
            )

            XCTAssertEqual(health.state, state)
            XCTAssertEqual(health.running, running)
            XCTAssertEqual(health.stable, stable)
            XCTAssertEqual(health.degraded, degraded)
            XCTAssertEqual(health.navigationUsable, navigationUsable)
            XCTAssertEqual(health.reasonMask, 0xA5)
            XCTAssertEqual(health.postInitTimeSec, 180.0, accuracy: 1.0e-6)
            XCTAssertEqual(health.distanceM, 750.0, accuracy: 1.0e-6)
            XCTAssertEqual(health.tailSamples, 30)
            XCTAssertEqual(health.mountSigmaMaxDeg, 1.5, accuracy: 1.0e-6)
            XCTAssertEqual(health.attitudeSigmaMaxDeg, 4.0, accuracy: 1.0e-6)
        }
    }

    func testSwiftFfiRawValuesMatchRustAbi() {
        XCTAssertEqual(HarshBehaviorPreset.sensitive.rawValue, SENSOR_FUSION_HARSH_BEHAVIOR_SENSITIVE)
        XCTAssertEqual(HarshBehaviorPreset.balanced.rawValue, SENSOR_FUSION_HARSH_BEHAVIOR_BALANCED)
        XCTAssertEqual(HarshBehaviorPreset.conservative.rawValue, SENSOR_FUSION_HARSH_BEHAVIOR_CONSERVATIVE)

        XCTAssertEqual(FusionGnssEvents.positionRejected.rawValue, SENSOR_FUSION_GNSS_EVENT_POSITION_REJECTED)
        XCTAssertEqual(FusionGnssEvents.velocityRejected.rawValue, SENSOR_FUSION_GNSS_EVENT_VELOCITY_REJECTED)
        XCTAssertEqual(
            FusionGnssEvents.positionConsecutiveRejected.rawValue,
            SENSOR_FUSION_GNSS_EVENT_POSITION_CONSECUTIVE_REJECTED
        )
        XCTAssertEqual(
            FusionGnssEvents.velocityConsecutiveRejected.rawValue,
            SENSOR_FUSION_GNSS_EVENT_VELOCITY_CONSECUTIVE_REJECTED
        )
        XCTAssertEqual(FusionGnssEvents.positionGapBypass.rawValue, SENSOR_FUSION_GNSS_EVENT_POSITION_GAP_BYPASS)
        XCTAssertEqual(FusionGnssEvents.velocityGapBypass.rawValue, SENSOR_FUSION_GNSS_EVENT_VELOCITY_GAP_BYPASS)
        XCTAssertEqual(
            FusionGnssEvents.positionAccuracyBypass.rawValue,
            SENSOR_FUSION_GNSS_EVENT_POSITION_ACCURACY_BYPASS
        )
        XCTAssertEqual(
            FusionGnssEvents.velocityAccuracyBypass.rawValue,
            SENSOR_FUSION_GNSS_EVENT_VELOCITY_ACCURACY_BYPASS
        )

        XCTAssertEqual(RoadEventDetection.Kind.harshAcceleration.rawValue, 1)
        XCTAssertEqual(RoadEventDetection.Kind.harshBraking.rawValue, 2)
        XCTAssertEqual(RoadEventDetection.Kind.harshCornering.rawValue, 3)
        XCTAssertEqual(RoadEventDetection.Kind.reverse.rawValue, 4)
        XCTAssertEqual(RoadEventDetection.Kind.speedBump.rawValue, 5)
        XCTAssertEqual(RoadEventDetection.Kind.uphill.rawValue, 6)
        XCTAssertEqual(RoadEventDetection.Kind.downhill.rawValue, 7)
        XCTAssertEqual(RoadEventDetection.Kind.roadShock.rawValue, 8)
        XCTAssertEqual(RoadEventDetection.Kind.roughRoad.rawValue, 9)
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
