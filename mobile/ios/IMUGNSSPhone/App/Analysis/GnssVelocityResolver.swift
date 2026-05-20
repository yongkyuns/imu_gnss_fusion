import Foundation

enum GnssVelocityResolver {
    static let stationarySpeedThresholdMps = 0.25
    static let defaultSpeedAccuracyMps = 5.0
    static let maxCourseAccuracyForHeadingDeg = 45.0

    static func horizontalVelocity(
        speedMps: Double?,
        courseDeg: Double?
    ) -> (northMps: Double, eastMps: Double)? {
        guard let speedMps, speedMps >= 0.0, speedMps.isFinite else {
            return nil
        }
        if speedMps <= stationarySpeedThresholdMps {
            return (0.0, 0.0)
        }
        guard let courseDeg, courseDeg >= 0.0, courseDeg.isFinite else {
            return nil
        }
        let headingRad = courseDeg * .pi / 180.0
        return (
            speedMps * cos(headingRad),
            speedMps * sin(headingRad)
        )
    }

    static func horizontalVelocityStdMps(
        speedMps: Double,
        speedAccuracyMps: Double?,
        courseAccuracyDeg: Double?
    ) -> Double {
        let speedStdMps = validPositive(speedAccuracyMps) ?? defaultSpeedAccuracyMps
        guard let courseAccuracyDeg = validNonNegative(courseAccuracyDeg) else {
            return speedStdMps
        }
        let directionStdMps = max(0.0, speedMps) * courseAccuracyDeg * .pi / 180.0
        return hypot(speedStdMps, directionStdMps)
    }

    static func headingRad(courseDeg: Double?, courseAccuracyDeg: Double?) -> Double? {
        guard let courseDeg, courseDeg >= 0.0, courseDeg.isFinite else { return nil }
        if let courseAccuracyDeg = validNonNegative(courseAccuracyDeg),
           courseAccuracyDeg > maxCourseAccuracyForHeadingDeg {
            return nil
        }
        return courseDeg * .pi / 180.0
    }

    private static func validPositive(_ value: Double?) -> Double? {
        guard let value, value.isFinite, value > 0.0 else { return nil }
        return value
    }

    private static func validNonNegative(_ value: Double?) -> Double? {
        guard let value, value.isFinite, value >= 0.0 else { return nil }
        return value
    }
}
