import Foundation

struct GnssFusionInput: Equatable, Sendable {
    let positionStdM: NavigationVectorNED
    let velocityNedMps: NavigationVectorNED
    let velocityStdMps: NavigationVectorNED
    let headingRad: Double?

    static func make(
        latitudeDeg: Double,
        longitudeDeg: Double,
        altitudeM: Double,
        velN: Double?,
        velE: Double?,
        velD: Double?,
        hAcc: Double,
        vAcc: Double,
        courseDeg: Double?,
        speedAccuracyMps: Double?,
        courseAccuracyDeg: Double?
    ) -> GnssFusionInput? {
        guard latitudeDeg.isFinite,
              longitudeDeg.isFinite,
              altitudeM.isFinite,
              (-90.0...90.0).contains(latitudeDeg),
              (-180.0...180.0).contains(longitudeDeg)
        else { return nil }

        guard let velN,
              let velE,
              velN.isFinite,
              velE.isFinite
        else { return nil }

        let horizontalStdM = hAcc.isFinite && hAcc > 0.0 ? hAcc : 25.0
        let verticalStdM = vAcc.isFinite && vAcc > 0.0 ? vAcc : 50.0
        let horizontalSpeedMps = hypot(velN, velE)
        if horizontalSpeedMps > GnssVelocityResolver.stationarySpeedThresholdMps,
           !GnssVelocityResolver.isCourseUsable(
               courseDeg: courseDeg,
               courseAccuracyDeg: courseAccuracyDeg
           ) {
            return nil
        }
        let horizontalVelocityStdMps = GnssVelocityResolver.horizontalVelocityStdMps(
            speedAccuracyMps: speedAccuracyMps
        )
        let velDValue = velD.map { $0.isFinite ? $0 : 0.0 } ?? 0.0
        let headingRad = GnssVelocityResolver.headingRad(
            courseDeg: courseDeg,
            courseAccuracyDeg: courseAccuracyDeg
        )

        return GnssFusionInput(
            positionStdM: NavigationVectorNED(
                north: horizontalStdM,
                east: horizontalStdM,
                down: verticalStdM
            ),
            velocityNedMps: NavigationVectorNED(
                north: velN,
                east: velE,
                down: velDValue
            ),
            velocityStdMps: NavigationVectorNED(
                north: horizontalVelocityStdMps,
                east: horizontalVelocityStdMps,
                down: 2.5
            ),
            headingRad: headingRad
        )
    }
}
