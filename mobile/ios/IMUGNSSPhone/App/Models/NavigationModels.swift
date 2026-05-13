import Foundation

struct NavigationVectorNED: Equatable, Sendable {
    var north: Double
    var east: Double
    var down: Double

    init(north: Double, east: Double, down: Double) {
        self.north = north
        self.east = east
        self.down = down
    }
}

struct VehicleVectorFRD: Equatable, Sendable {
    var forward: Double
    var right: Double
    var down: Double

    init(forward: Double, right: Double, down: Double) {
        self.forward = forward
        self.right = right
        self.down = down
    }
}

struct GeographicCoordinate: Equatable, Sendable {
    var latitudeDeg: Double
    var longitudeDeg: Double
    var altitudeM: Double?

    init(latitudeDeg: Double, longitudeDeg: Double, altitudeM: Double? = nil) {
        self.latitudeDeg = latitudeDeg
        self.longitudeDeg = longitudeDeg
        self.altitudeM = altitudeM
    }

    var isValidLatitudeLongitude: Bool {
        latitudeDeg.isFinite
            && longitudeDeg.isFinite
            && (-90.0 ... 90.0).contains(latitudeDeg)
            && (-180.0 ... 180.0).contains(longitudeDeg)
    }
}

struct RoutePoint: Equatable, Identifiable, Sendable {
    enum Source: String, Equatable, Sendable {
        case gnss
        case fused
    }

    var id: String
    var timestamp: Date
    var source: Source
    var coordinate: GeographicCoordinate
    var positionNEDM: NavigationVectorNED?
    var horizontalAccuracyM: Double?
    var verticalAccuracyM: Double?

    init(
        id: String? = nil,
        timestamp: Date,
        source: Source,
        coordinate: GeographicCoordinate,
        positionNEDM: NavigationVectorNED? = nil,
        horizontalAccuracyM: Double? = nil,
        verticalAccuracyM: Double? = nil
    ) {
        self.timestamp = timestamp
        self.source = source
        self.coordinate = coordinate
        self.positionNEDM = positionNEDM
        self.horizontalAccuracyM = Self.validAccuracy(horizontalAccuracyM)
        self.verticalAccuracyM = Self.validAccuracy(verticalAccuracyM)
        self.id = id ?? "\(source.rawValue)-\(timestamp.timeIntervalSince1970)"
    }

    private static func validAccuracy(_ value: Double?) -> Double? {
        guard let value, value.isFinite, value >= 0.0 else { return nil }
        return value
    }
}

struct GnssRouteSample: Equatable, Sendable {
    var point: RoutePoint
    var velocityNEDMps: NavigationVectorNED?
    var speedMps: Double?
    var courseDeg: Double?
    var speedAccuracyMps: Double?

    init(
        timestamp: Date,
        coordinate: GeographicCoordinate,
        positionNEDM: NavigationVectorNED? = nil,
        velocityNEDMps: NavigationVectorNED? = nil,
        speedMps: Double? = nil,
        courseDeg: Double? = nil,
        horizontalAccuracyM: Double? = nil,
        verticalAccuracyM: Double? = nil,
        speedAccuracyMps: Double? = nil
    ) {
        self.point = RoutePoint(
            timestamp: timestamp,
            source: .gnss,
            coordinate: coordinate,
            positionNEDM: positionNEDM,
            horizontalAccuracyM: horizontalAccuracyM,
            verticalAccuracyM: verticalAccuracyM
        )
        self.velocityNEDMps = velocityNEDMps
        self.speedMps = Self.validNonNegative(speedMps)
        self.courseDeg = Self.validCourse(courseDeg)
        self.speedAccuracyMps = Self.validNonNegative(speedAccuracyMps)
    }

    var hasUsableVelocity: Bool {
        velocityNEDMps != nil || (speedMps != nil && courseDeg != nil)
    }

    func hasUsableAccuracy(maxHorizontalAccuracyM: Double) -> Bool {
        guard let horizontalAccuracyM = point.horizontalAccuracyM else { return false }
        return horizontalAccuracyM <= maxHorizontalAccuracyM
    }

    private static func validNonNegative(_ value: Double?) -> Double? {
        guard let value, value.isFinite, value >= 0.0 else { return nil }
        return value
    }

    private static func validCourse(_ value: Double?) -> Double? {
        guard let value, value.isFinite, (0.0 ..< 360.0).contains(value) else { return nil }
        return value
    }
}

struct FusedRouteSample: Equatable, Sendable {
    var point: RoutePoint
    var velocityNEDMps: NavigationVectorNED
    var attitudeQNV: Quaternion?
    var health: FusionHealth

    init(
        timestamp: Date,
        coordinate: GeographicCoordinate,
        positionNEDM: NavigationVectorNED? = nil,
        velocityNEDMps: NavigationVectorNED,
        attitudeQNV: Quaternion? = nil,
        health: FusionHealth
    ) {
        self.point = RoutePoint(
            timestamp: timestamp,
            source: .fused,
            coordinate: coordinate,
            positionNEDM: positionNEDM
        )
        self.velocityNEDMps = velocityNEDMps
        self.attitudeQNV = attitudeQNV
        self.health = health
    }
}

enum RouteLayerSelection: String, CaseIterable, Equatable, Identifiable, Sendable {
    case none
    case fused
    case gnss
    case both
    case delta

    var id: String { rawValue }

    var showsFusedRoute: Bool {
        self == .fused || self == .both || self == .delta
    }

    var showsGnssRoute: Bool {
        self == .gnss || self == .both || self == .delta
    }

    var showsDeltaOverlay: Bool {
        self == .delta
    }

    func togglingFusedRoute() -> RouteLayerSelection {
        Self.selection(fused: !showsFusedRoute, gnss: showsGnssRoute)
    }

    func togglingGnssRoute() -> RouteLayerSelection {
        Self.selection(fused: showsFusedRoute, gnss: !showsGnssRoute)
    }

    private static func selection(fused: Bool, gnss: Bool) -> RouteLayerSelection {
        switch (fused, gnss) {
        case (true, true):
            return .both
        case (true, false):
            return .fused
        case (false, true):
            return .gnss
        case (false, false):
            return .none
        }
    }
}
