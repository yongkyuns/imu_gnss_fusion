import Foundation

#if DEBUG
struct DeveloperComparisonSnapshot: Equatable, Sendable {
    let positionErrorM: NavigationVectorNED?
    let horizontalPositionErrorM: Double?
    let velocityErrorMps: NavigationVectorNED?
    let groundSpeedErrorMps: Double?
    let courseErrorDeg: Double?
    let attitudeErrorDeg: NavigationVectorNED?
}

enum DeveloperComparison {
    static func makeSnapshot(
        iosPositionM: NavigationVectorNED?,
        fusedPositionM: NavigationVectorNED?,
        iosVelocityMps: NavigationVectorNED?,
        fusedVelocityMps: NavigationVectorNED?,
        iosCourseDeg: Double?,
        fusedYawDeg: Double?,
        iosAttitudeEulerDeg: NavigationVectorNED?,
        fusedEulerDeg: NavigationVectorNED?
    ) -> DeveloperComparisonSnapshot {
        let positionError = difference(estimate: fusedPositionM, reference: iosPositionM)
        let velocityError = difference(estimate: fusedVelocityMps, reference: iosVelocityMps)
        return DeveloperComparisonSnapshot(
            positionErrorM: positionError,
            horizontalPositionErrorM: positionError.map(horizontalMagnitude),
            velocityErrorMps: velocityError,
            groundSpeedErrorMps: groundSpeedError(estimate: fusedVelocityMps, reference: iosVelocityMps),
            courseErrorDeg: angularErrorDeg(estimate: fusedYawDeg, reference: iosCourseDeg),
            attitudeErrorDeg: eulerErrorDeg(estimate: fusedEulerDeg, reference: iosAttitudeEulerDeg)
        )
    }

    static func difference(
        estimate: NavigationVectorNED?,
        reference: NavigationVectorNED?
    ) -> NavigationVectorNED? {
        guard let estimate, let reference else { return nil }
        return NavigationVectorNED(
            north: estimate.north - reference.north,
            east: estimate.east - reference.east,
            down: estimate.down - reference.down
        )
    }

    static func horizontalMagnitude(_ vector: NavigationVectorNED) -> Double {
        hypot(vector.north, vector.east)
    }

    static func groundSpeedError(
        estimate: NavigationVectorNED?,
        reference: NavigationVectorNED?
    ) -> Double? {
        guard let estimate, let reference else { return nil }
        return horizontalMagnitude(estimate) - horizontalMagnitude(reference)
    }

    static func angularErrorDeg(estimate: Double?, reference: Double?) -> Double? {
        guard let estimate, let reference else { return nil }
        var delta = estimate - reference
        while delta > 180.0 {
            delta -= 360.0
        }
        while delta < -180.0 {
            delta += 360.0
        }
        return delta
    }

    static func eulerErrorDeg(
        estimate: NavigationVectorNED?,
        reference: NavigationVectorNED?
    ) -> NavigationVectorNED? {
        guard let estimate, let reference else { return nil }
        return NavigationVectorNED(
            north: angularErrorDeg(estimate: estimate.north, reference: reference.north) ?? .nan,
            east: angularErrorDeg(estimate: estimate.east, reference: reference.east) ?? .nan,
            down: angularErrorDeg(estimate: estimate.down, reference: reference.down) ?? .nan
        )
    }
}
#endif
