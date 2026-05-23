import Foundation

enum GnssVelocityResolver {
    static let stationarySpeedThresholdMps = 0.25
    static let defaultSpeedAccuracyMps = 5.0
    static let maxCourseAccuracyForHeadingDeg = 45.0

    static func horizontalVelocity(
        speedMps: Double?,
        courseDeg: Double?,
        courseAccuracyDeg: Double?
    ) -> (northMps: Double, eastMps: Double)? {
        guard let speedMps, speedMps >= 0.0, speedMps.isFinite else {
            return nil
        }
        if speedMps <= stationarySpeedThresholdMps {
            return (0.0, 0.0)
        }
        guard isCourseUsable(courseDeg: courseDeg, courseAccuracyDeg: courseAccuracyDeg),
              let courseDeg
        else {
            return nil
        }
        let headingRad = courseDeg * .pi / 180.0
        return (
            speedMps * cos(headingRad),
            speedMps * sin(headingRad)
        )
    }

    static func horizontalVelocityStdMps(
        speedAccuracyMps: Double?
    ) -> Double {
        validPositive(speedAccuracyMps) ?? defaultSpeedAccuracyMps
    }

    static func headingRad(courseDeg: Double?, courseAccuracyDeg: Double?) -> Double? {
        guard isCourseUsable(courseDeg: courseDeg, courseAccuracyDeg: courseAccuracyDeg),
              let courseDeg
        else {
            return nil
        }
        return courseDeg * .pi / 180.0
    }

    static func isCourseUsable(courseDeg: Double?, courseAccuracyDeg: Double?) -> Bool {
        guard let courseDeg, courseDeg >= 0.0, courseDeg.isFinite else { return false }
        guard let courseAccuracyDeg = validNonNegative(courseAccuracyDeg) else { return true }
        return courseAccuracyDeg <= maxCourseAccuracyForHeadingDeg
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
