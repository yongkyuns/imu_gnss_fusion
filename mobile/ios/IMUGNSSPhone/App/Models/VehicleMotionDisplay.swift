import Foundation

struct VehicleMotionDisplay: Equatable, Sendable {
    enum Segment: String, Equatable, Sendable {
        case stopped
        case straight
        case accelerating
        case braking
        case turning
        case roughRoad
        case gnssDegraded

        var displayTitle: String {
            switch self {
            case .stopped: return "Stopped"
            case .straight: return "Straight"
            case .accelerating: return "Accelerating"
            case .braking: return "Braking"
            case .turning: return "Turning"
            case .roughRoad: return "Rough Road"
            case .gnssDegraded: return "GNSS Degraded"
            }
        }
    }

    var vehicleVelocityFRDMps: VehicleVectorFRD
    var groundSpeedMps: Double
    var speed3DMps: Double
    var yawRateRadps: Double?
    var curvaturePerM: Double?
    var sideslipRad: Double?
    var segment: Segment

    init(
        vehicleVelocityFRDMps: VehicleVectorFRD,
        groundSpeedMps: Double,
        speed3DMps: Double,
        yawRateRadps: Double? = nil,
        curvaturePerM: Double? = nil,
        sideslipRad: Double? = nil,
        segment: Segment
    ) {
        self.vehicleVelocityFRDMps = vehicleVelocityFRDMps
        self.groundSpeedMps = groundSpeedMps
        self.speed3DMps = speed3DMps
        self.yawRateRadps = yawRateRadps
        self.curvaturePerM = curvaturePerM
        self.sideslipRad = sideslipRad
        self.segment = segment
    }

    static func make(
        nedVelocityMps: NavigationVectorNED,
        attitudeQNV: Quaternion,
        yawRateRadps: Double? = nil,
        longitudinalAccelerationMps2: Double? = nil,
        verticalAccelerationMps2: Double? = nil,
        health: FusionHealth,
        stoppedSpeedThresholdMps: Double = 0.5,
        sideslipMinimumForwardSpeedMps: Double = 2.0
    ) -> VehicleMotionDisplay {
        let vehicleVelocity = MotionKinematics.vehicleFRDVelocity(
            qNV: attitudeQNV,
            nedVelocityMps: nedVelocityMps
        )
        let groundSpeed = MotionKinematics.groundSpeed(nedVelocityMps)
        let speed3D = MotionKinematics.speed3D(nedVelocityMps)
        let sideslip = MotionKinematics.sideslipAngleRad(
            vehicleVelocityFRDMps: vehicleVelocity,
            minimumForwardSpeedMps: sideslipMinimumForwardSpeedMps
        )
        let curvature = MotionKinematics.curvaturePerM(
            yawRateRadps: yawRateRadps,
            groundSpeedMps: groundSpeed
        )
        let segment = classifySegment(
            health: health,
            groundSpeedMps: groundSpeed,
            yawRateRadps: yawRateRadps,
            longitudinalAccelerationMps2: longitudinalAccelerationMps2,
            verticalAccelerationMps2: verticalAccelerationMps2,
            stoppedSpeedThresholdMps: stoppedSpeedThresholdMps
        )

        return VehicleMotionDisplay(
            vehicleVelocityFRDMps: vehicleVelocity,
            groundSpeedMps: groundSpeed,
            speed3DMps: speed3D,
            yawRateRadps: yawRateRadps,
            curvaturePerM: curvature,
            sideslipRad: sideslip,
            segment: segment
        )
    }

    private static func classifySegment(
        health: FusionHealth,
        groundSpeedMps: Double,
        yawRateRadps: Double?,
        longitudinalAccelerationMps2: Double?,
        verticalAccelerationMps2: Double?,
        stoppedSpeedThresholdMps: Double
    ) -> Segment {
        if health.status == .poorGNSS {
            return .gnssDegraded
        }
        if groundSpeedMps < stoppedSpeedThresholdMps {
            return .stopped
        }
        if let verticalAccelerationMps2, abs(verticalAccelerationMps2) > 2.5 {
            return .roughRoad
        }
        if let yawRateRadps, abs(yawRateRadps) > 0.12 {
            return .turning
        }
        if let longitudinalAccelerationMps2 {
            if longitudinalAccelerationMps2 > 0.8 {
                return .accelerating
            }
            if longitudinalAccelerationMps2 < -0.8 {
                return .braking
            }
        }
        return .straight
    }
}
