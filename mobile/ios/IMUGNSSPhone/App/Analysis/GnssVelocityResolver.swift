import Foundation

enum GnssVelocityResolver {
    static let stationarySpeedThresholdMps = 0.25

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
}
