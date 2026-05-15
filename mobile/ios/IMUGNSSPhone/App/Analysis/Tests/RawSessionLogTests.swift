import XCTest
@testable import IMUGNSSPhone

final class RawSessionLogTests: XCTestCase {
    func testRawSessionLogRoundTripsAndSummarizesEventCounts() throws {
        let start = Date(timeIntervalSince1970: 1_700_000_000)
        let log = RawSessionLog(
            id: UUID(uuidString: "A8D2B956-1DF0-4C96-9143-42FE8B1818B4")!,
            name: "Fixture Drive",
            startTime: start,
            appVersion: "test",
            buildNumber: "1",
            events: [
                .imu(
                    RawImuSample(
                        sourceUptimeSec: 10.0,
                        accelXMps2: 1.0,
                        accelYMps2: 2.0,
                        accelZMps2: 3.0,
                        gyroXRadps: 0.1,
                        gyroYRadps: 0.2,
                        gyroZRadps: 0.3,
                        attitudeReferenceFrame: "default"
                    ),
                    elapsedSec: 0.1,
                    wallTime: start.addingTimeInterval(0.1)
                ),
                .gnss(
                    RawGnssSample(
                        latitudeDeg: 37.0,
                        longitudeDeg: -122.0,
                        altitudeM: 8.0,
                        horizontalAccuracyM: 4.0,
                        verticalAccuracyM: 6.0,
                        speedMps: 12.0,
                        courseDeg: 90.0,
                        speedAccuracyMps: 0.5,
                        courseAccuracyDeg: 1.0,
                        positionNorthM: 1.0,
                        positionEastM: 2.0,
                        positionDownM: -0.5,
                        velocityNorthMps: 0.0,
                        velocityEastMps: 12.0,
                        velocityDownMps: 0.1
                    ),
                    elapsedSec: 0.5,
                    wallTime: start.addingTimeInterval(0.5)
                ),
                .barometer(
                    RawBarometerSample(
                        sourceUptimeSec: 10.2,
                        relativeAltitudeM: 0.3,
                        pressureKPa: 101.2,
                        derivedVerticalVelocityDownMps: -0.1
                    ),
                    elapsedSec: 0.3,
                    wallTime: start.addingTimeInterval(0.3)
                )
            ]
        )

        let data = try RawSessionJSON.makeEncoder().encode(log)
        let decoded = try RawSessionJSON.makeDecoder().decode(RawSessionLog.self, from: data)

        XCTAssertEqual(decoded.id, log.id)
        XCTAssertEqual(decoded.name, log.name)
        XCTAssertEqual(decoded.startTime.timeIntervalSince1970, log.startTime.timeIntervalSince1970, accuracy: 1e-6)
        XCTAssertEqual(decoded.events.count, log.events.count)
        XCTAssertEqual(decoded.events[0].wallTime?.timeIntervalSince(log.startTime) ?? .nan, 0.1, accuracy: 1e-6)
        XCTAssertEqual(decoded.durationSec, 0.5, accuracy: 1e-12)
        XCTAssertEqual(decoded.summary().imuCount, 1)
        XCTAssertEqual(decoded.summary().gnssCount, 1)
        XCTAssertEqual(decoded.summary().barometerCount, 1)
    }

    func testSummaryWithoutFileURLRepresentsPendingSave() {
        let pending = RawSessionSummary(
            id: UUID(uuidString: "61F52895-D66D-478C-9B54-149DA8AD5A6C")!,
            name: "Saving Drive",
            startTime: Date(timeIntervalSince1970: 1_700_000_000),
            durationSec: 12.0,
            imuCount: 100,
            gnssCount: 12,
            barometerCount: 12,
            fileURL: nil
        )
        let saved = RawSessionSummary(
            id: pending.id,
            name: pending.name,
            startTime: pending.startTime,
            durationSec: pending.durationSec,
            imuCount: pending.imuCount,
            gnssCount: pending.gnssCount,
            barometerCount: pending.barometerCount,
            fileURL: URL(fileURLWithPath: "/tmp/saved.motionfusion")
        )

        XCTAssertTrue(pending.isPendingSave)
        XCTAssertFalse(saved.isPendingSave)
    }

    func testTimelineValidatesEnvelopePayloads() {
        let invalid = RawSensorEventEnvelope(
            kind: .imu,
            elapsedSec: 0.0,
            wallTime: nil,
            imu: nil,
            gnss: nil,
            barometer: nil
        )
        let log = RawSessionLog(name: "Invalid", startTime: Date(timeIntervalSince1970: 0), events: [invalid])

        XCTAssertThrowsError(try RawSessionTimeline.events(for: log)) { error in
            XCTAssertEqual(error as? RawSessionValidationError, .missingPayload("imu"))
        }
    }

    func testTimelineOrdersEventsByElapsedTimeThenSensorPriority() throws {
        let start = Date(timeIntervalSince1970: 0)
        let imu = RawSensorEventEnvelope.imu(
            RawImuSample(
                sourceUptimeSec: nil,
                accelXMps2: 0.0,
                accelYMps2: 0.0,
                accelZMps2: 9.8,
                gyroXRadps: 0.0,
                gyroYRadps: 0.0,
                gyroZRadps: 0.0,
                attitudeReferenceFrame: nil
            ),
            elapsedSec: 1.0,
            wallTime: nil
        )
        let gnss = RawSensorEventEnvelope.gnss(
            RawGnssSample(
                latitudeDeg: 37.0,
                longitudeDeg: -122.0,
                altitudeM: 0.0,
                horizontalAccuracyM: nil,
                verticalAccuracyM: nil,
                speedMps: nil,
                courseDeg: nil,
                speedAccuracyMps: nil,
                courseAccuracyDeg: nil,
                positionNorthM: nil,
                positionEastM: nil,
                positionDownM: nil,
                velocityNorthMps: nil,
                velocityEastMps: nil,
                velocityDownMps: nil
            ),
            elapsedSec: 1.0,
            wallTime: nil
        )
        let barometer = RawSensorEventEnvelope.barometer(
            RawBarometerSample(
                sourceUptimeSec: nil,
                relativeAltitudeM: 0.0,
                pressureKPa: nil,
                derivedVerticalVelocityDownMps: nil
            ),
            elapsedSec: 1.0,
            wallTime: nil
        )
        let earlierGnss = RawSensorEventEnvelope.gnss(
            gnss.gnss!,
            elapsedSec: 0.5,
            wallTime: nil
        )
        let log = RawSessionLog(name: "Ordered", startTime: start, events: [gnss, imu, earlierGnss, barometer])

        let events = try RawSessionTimeline.events(for: log)

        XCTAssertEqual(events.map(\.elapsedSec), [0.5, 1.0, 1.0, 1.0])
        guard case .gnss = events[0],
              case .barometer = events[1],
              case .imu = events[2],
              case .gnss = events[3]
        else {
            return XCTFail("Unexpected replay ordering")
        }
    }

    func testFileStoreSavesLoadsSummariesAndDeletesSessions() throws {
        let rootURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("RawSessionLogTests-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: rootURL)
        }
        let store = RawSessionFileStore(rootURL: rootURL)
        let log = RawSessionLog(
            name: "Store Fixture",
            startTime: Date(timeIntervalSince1970: 42),
            events: [
                .barometer(
                    RawBarometerSample(
                        sourceUptimeSec: 1.0,
                        relativeAltitudeM: 2.0,
                        pressureKPa: 100.0,
                        derivedVerticalVelocityDownMps: 0.0
                    ),
                    elapsedSec: 2.5,
                    wallTime: nil
                )
            ]
        )

        let url = try store.save(log)
        let loaded = try store.load(from: url)
        let summaries = try store.summaries()

        XCTAssertEqual(loaded, log)
        XCTAssertEqual(summaries.count, 1)
        XCTAssertEqual(summaries[0].name, "Store Fixture")
        XCTAssertEqual(summaries[0].durationSec, 2.5, accuracy: 1e-12)

        try store.delete(summaries[0])
        XCTAssertEqual(try store.summaries(), [])
    }

    func testFileStoreOverwritesSameSessionForCheckpoints() throws {
        let rootURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("RawSessionCheckpointTests-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: rootURL)
        }
        let store = RawSessionFileStore(rootURL: rootURL)
        let id = UUID(uuidString: "D23F5A1D-BD28-4B8D-91E6-9CF49C26968B")!
        let start = Date(timeIntervalSince1970: 100)
        let first = RawSessionLog(
            id: id,
            name: "Checkpoint Fixture",
            startTime: start,
            events: [
                .barometer(
                    RawBarometerSample(
                        sourceUptimeSec: 1.0,
                        relativeAltitudeM: 1.0,
                        pressureKPa: nil,
                        derivedVerticalVelocityDownMps: nil
                    ),
                    elapsedSec: 1.0,
                    wallTime: nil
                )
            ]
        )
        var second = first
        second.events.append(
            .barometer(
                RawBarometerSample(
                    sourceUptimeSec: 2.0,
                    relativeAltitudeM: 2.0,
                    pressureKPa: nil,
                    derivedVerticalVelocityDownMps: nil
                ),
                elapsedSec: 2.0,
                wallTime: nil
            )
        )

        let firstURL = try store.save(first)
        let secondURL = try store.save(second)
        let loaded = try store.load(from: secondURL)
        let summaries = try store.summaries()

        XCTAssertEqual(firstURL, secondURL)
        XCTAssertEqual(loaded.events.count, 2)
        XCTAssertEqual(summaries.count, 1)
        XCTAssertEqual(summaries[0].durationSec, 2.0, accuracy: 1e-12)
    }
}
