import Foundation

struct Quaternion: Equatable, Sendable {
    var w: Double
    var x: Double
    var y: Double
    var z: Double

    init(w: Double, x: Double, y: Double, z: Double) {
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    }

    static let identity = Quaternion(w: 1.0, x: 0.0, y: 0.0, z: 0.0)

    var norm: Double {
        sqrt(w * w + x * x + y * y + z * z)
    }

    var normalized: Quaternion {
        let n = norm
        guard n.isFinite, n > 0.0 else { return .identity }
        return Quaternion(w: w / n, x: x / n, y: y / n, z: z / n)
    }

    var conjugated: Quaternion {
        Quaternion(w: w, x: -x, y: -y, z: -z)
    }

    func multiplied(by rhs: Quaternion) -> Quaternion {
        Quaternion(
            w: w * rhs.w - x * rhs.x - y * rhs.y - z * rhs.z,
            x: w * rhs.x + x * rhs.w + y * rhs.z - z * rhs.y,
            y: w * rhs.y - x * rhs.z + y * rhs.w + z * rhs.x,
            z: w * rhs.z + x * rhs.y - y * rhs.x + z * rhs.w
        )
    }

    func rotated(_ vector: NavigationVectorNED) -> NavigationVectorNED {
        let q = normalized
        let v = Quaternion(w: 0.0, x: vector.north, y: vector.east, z: vector.down)
        let rotated = q.multiplied(by: v).multiplied(by: q.conjugated)
        return NavigationVectorNED(north: rotated.x, east: rotated.y, down: rotated.z)
    }

    func rotated(x: Double, y: Double, z: Double) -> (x: Double, y: Double, z: Double) {
        let q = normalized
        let v = Quaternion(w: 0.0, x: x, y: y, z: z)
        let rotated = q.multiplied(by: v).multiplied(by: q.conjugated)
        return (rotated.x, rotated.y, rotated.z)
    }
}

enum MotionKinematics {
    static func groundSpeed(_ nedVelocityMps: NavigationVectorNED) -> Double {
        hypot(nedVelocityMps.north, nedVelocityMps.east)
    }

    static func speed3D(_ nedVelocityMps: NavigationVectorNED) -> Double {
        sqrt(
            nedVelocityMps.north * nedVelocityMps.north
                + nedVelocityMps.east * nedVelocityMps.east
                + nedVelocityMps.down * nedVelocityMps.down
        )
    }

    /// Converts navigation-frame NED velocity into vehicle-frame FRD velocity.
    ///
    /// `qNV` is the scalar-first quaternion that rotates vehicle-frame vectors
    /// into navigation frame. The inverse rotation maps NED velocity into
    /// forward/right/down vehicle display axes.
    static func vehicleFRDVelocity(
        qNV: Quaternion,
        nedVelocityMps: NavigationVectorNED
    ) -> VehicleVectorFRD {
        let qVN = qNV.normalized.conjugated
        let vehicle = qVN.rotated(nedVelocityMps)
        return VehicleVectorFRD(
            forward: vehicle.north,
            right: vehicle.east,
            down: vehicle.down
        )
    }

    /// Converts body-frame angular rate into vehicle-frame FRD angular rate.
    ///
    /// `qBV` rotates vehicle-frame vectors into body frame. The inverse
    /// rotation maps body-frame gyroscope readings into vehicle axes.
    static func vehicleFRDGyro(
        qBV: Quaternion,
        bodyGyroRadps: (x: Double, y: Double, z: Double)
    ) -> VehicleVectorFRD {
        let vehicle = qBV.normalized.conjugated.rotated(
            x: bodyGyroRadps.x,
            y: bodyGyroRadps.y,
            z: bodyGyroRadps.z
        )
        return VehicleVectorFRD(
            forward: vehicle.x,
            right: vehicle.y,
            down: vehicle.z
        )
    }

    /// Converts body-frame specific force into vehicle-frame FRD axes.
    static func vehicleFRDAcceleration(
        qBV: Quaternion,
        bodyAccelMps2: (x: Double, y: Double, z: Double)
    ) -> VehicleVectorFRD {
        let vehicle = qBV.normalized.conjugated.rotated(
            x: bodyAccelMps2.x,
            y: bodyAccelMps2.y,
            z: bodyAccelMps2.z
        )
        return VehicleVectorFRD(
            forward: vehicle.x,
            right: vehicle.y,
            down: vehicle.z
        )
    }

    static func sideslipAngleRad(
        vehicleVelocityFRDMps: VehicleVectorFRD,
        minimumForwardSpeedMps: Double = 2.0
    ) -> Double? {
        guard abs(vehicleVelocityFRDMps.forward) >= minimumForwardSpeedMps else {
            return nil
        }
        return atan2(vehicleVelocityFRDMps.right, vehicleVelocityFRDMps.forward)
    }

    static func curvaturePerM(
        yawRateRadps: Double?,
        groundSpeedMps: Double,
        minimumGroundSpeedMps: Double = 1.0
    ) -> Double? {
        guard let yawRateRadps, groundSpeedMps >= minimumGroundSpeedMps else {
            return nil
        }
        return yawRateRadps / groundSpeedMps
    }
}
