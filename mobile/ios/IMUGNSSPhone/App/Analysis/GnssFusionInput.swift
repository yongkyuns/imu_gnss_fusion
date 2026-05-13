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
        speedAccuracyMps: Double?
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
        let speedStdMps = speedAccuracyMps.map { $0.isFinite && $0 > 0.0 ? $0 : 5.0 } ?? 5.0
        let velDValue = velD.map { $0.isFinite ? $0 : 0.0 } ?? 0.0
        let headingRad = courseDeg.flatMap { course -> Double? in
            guard course.isFinite, course >= 0.0 else { return nil }
            return course * .pi / 180.0
        }

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
                north: speedStdMps,
                east: speedStdMps,
                down: 2.5
            ),
            headingRad: headingRad
        )
    }
}
