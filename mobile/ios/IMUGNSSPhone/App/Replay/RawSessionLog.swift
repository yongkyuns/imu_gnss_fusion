import Foundation

struct RawSessionSummary: Identifiable, Equatable, Sendable {
    let id: UUID
    let name: String
    let startTime: Date
    let durationSec: Double
    let imuCount: Int
    let gnssCount: Int
    let barometerCount: Int
    let fileURL: URL?
}

struct RawSessionLog: Codable, Equatable, Sendable {
    static let schemaVersion = 1

    var schemaVersion: Int = Self.schemaVersion
    var id: UUID
    var name: String
    var startTime: Date
    var appVersion: String
    var buildNumber: String
    var events: [RawSensorEventEnvelope]

    init(
        id: UUID = UUID(),
        name: String,
        startTime: Date,
        appVersion: String = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0",
        buildNumber: String = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "0",
        events: [RawSensorEventEnvelope] = []
    ) {
        self.id = id
        self.name = name
        self.startTime = startTime
        self.appVersion = appVersion
        self.buildNumber = buildNumber
        self.events = events
    }

    var durationSec: Double {
        max(0.0, events.map(\.elapsedSec).max() ?? 0.0)
    }

    var imuCount: Int {
        events.filter { $0.kind == .imu }.count
    }

    var gnssCount: Int {
        events.filter { $0.kind == .gnss }.count
    }

    var barometerCount: Int {
        events.filter { $0.kind == .barometer }.count
    }

    func summary(fileURL: URL? = nil) -> RawSessionSummary {
        RawSessionSummary(
            id: id,
            name: name,
            startTime: startTime,
            durationSec: durationSec,
            imuCount: imuCount,
            gnssCount: gnssCount,
            barometerCount: barometerCount,
            fileURL: fileURL
        )
    }
}

struct RawSensorEventEnvelope: Codable, Equatable, Sendable {
    enum Kind: String, Codable, Equatable, Sendable {
        case imu
        case gnss
        case barometer
    }

    var schemaVersion: Int = RawSessionLog.schemaVersion
    let kind: Kind
    let elapsedSec: Double
    let wallTime: Date?
    let imu: RawImuSample?
    let gnss: RawGnssSample?
    let barometer: RawBarometerSample?

    static func imu(_ sample: RawImuSample, elapsedSec: Double, wallTime: Date?) -> RawSensorEventEnvelope {
        RawSensorEventEnvelope(kind: .imu, elapsedSec: elapsedSec, wallTime: wallTime, imu: sample, gnss: nil, barometer: nil)
    }

    static func gnss(_ sample: RawGnssSample, elapsedSec: Double, wallTime: Date?) -> RawSensorEventEnvelope {
        RawSensorEventEnvelope(kind: .gnss, elapsedSec: elapsedSec, wallTime: wallTime, imu: nil, gnss: sample, barometer: nil)
    }

    static func barometer(_ sample: RawBarometerSample, elapsedSec: Double, wallTime: Date?) -> RawSensorEventEnvelope {
        RawSensorEventEnvelope(kind: .barometer, elapsedSec: elapsedSec, wallTime: wallTime, imu: nil, gnss: nil, barometer: sample)
    }

    func validate() throws {
        guard schemaVersion == RawSessionLog.schemaVersion else {
            throw RawSessionValidationError.unsupportedSchema(schemaVersion)
        }
        guard elapsedSec.isFinite, elapsedSec >= 0.0 else {
            throw RawSessionValidationError.invalidElapsedTime(elapsedSec)
        }

        switch kind {
        case .imu where imu == nil:
            throw RawSessionValidationError.missingPayload(kind.rawValue)
        case .gnss where gnss == nil:
            throw RawSessionValidationError.missingPayload(kind.rawValue)
        case .barometer where barometer == nil:
            throw RawSessionValidationError.missingPayload(kind.rawValue)
        default:
            break
        }
    }
}

struct RawImuSample: Codable, Equatable, Sendable {
    let sourceUptimeSec: Double?
    let accelXMps2: Double
    let accelYMps2: Double
    let accelZMps2: Double
    let gyroXRadps: Double
    let gyroYRadps: Double
    let gyroZRadps: Double
    let attitudeReferenceFrame: String?
    var attitudeRollRad: Double? = nil
    var attitudePitchRad: Double? = nil
    var attitudeYawRad: Double? = nil
}

struct RawGnssSample: Codable, Equatable, Sendable {
    let latitudeDeg: Double
    let longitudeDeg: Double
    let altitudeM: Double
    let horizontalAccuracyM: Double?
    let verticalAccuracyM: Double?
    let speedMps: Double?
    let courseDeg: Double?
    let speedAccuracyMps: Double?
    let courseAccuracyDeg: Double?
    let positionNorthM: Double?
    let positionEastM: Double?
    let positionDownM: Double?
    let velocityNorthMps: Double?
    let velocityEastMps: Double?
    let velocityDownMps: Double?
}

struct RawBarometerSample: Codable, Equatable, Sendable {
    let sourceUptimeSec: Double?
    let relativeAltitudeM: Double
    let pressureKPa: Double?
    let derivedVerticalVelocityDownMps: Double?
}

enum RawReplayEvent: Equatable, Sendable {
    case imu(RawSensorEventEnvelope, RawImuSample)
    case gnss(RawSensorEventEnvelope, RawGnssSample)
    case barometer(RawSensorEventEnvelope, RawBarometerSample)

    var elapsedSec: Double {
        switch self {
        case .imu(let envelope, _): return envelope.elapsedSec
        case .gnss(let envelope, _): return envelope.elapsedSec
        case .barometer(let envelope, _): return envelope.elapsedSec
        }
    }
}

enum RawSessionTimeline {
    static func events(for log: RawSessionLog) throws -> [RawReplayEvent] {
        let events = try log.events.map { envelope -> RawReplayEvent in
            try envelope.validate()
            switch envelope.kind {
            case .imu:
                return .imu(envelope, envelope.imu!)
            case .gnss:
                return .gnss(envelope, envelope.gnss!)
            case .barometer:
                return .barometer(envelope, envelope.barometer!)
            }
        }
        return events.sorted { lhs, rhs in
            if lhs.elapsedSec == rhs.elapsedSec {
                return priority(lhs) < priority(rhs)
            }
            return lhs.elapsedSec < rhs.elapsedSec
        }
    }

    private static func priority(_ event: RawReplayEvent) -> Int {
        switch event {
        case .barometer: return 0
        case .imu: return 1
        case .gnss: return 2
        }
    }
}

enum RawSessionValidationError: Error, Equatable {
    case unsupportedSchema(Int)
    case invalidElapsedTime(Double)
    case missingPayload(String)
}
