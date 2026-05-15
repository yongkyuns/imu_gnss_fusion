import XCTest
@testable import IMUGNSSPhone

final class FusionTimingPolicyTests: XCTestCase {
    func testGnssRequiresRecentImuSample() {
        let gnssDate = Date(timeIntervalSince1970: 100.0)

        XCTAssertFalse(FusionTimingPolicy.hasFreshImu(
            for: gnssDate,
            lastImuSampleDate: nil
        ))
        XCTAssertTrue(FusionTimingPolicy.hasFreshImu(
            for: gnssDate,
            lastImuSampleDate: Date(timeIntervalSince1970: 99.5)
        ))
        XCTAssertTrue(FusionTimingPolicy.hasFreshImu(
            for: gnssDate,
            lastImuSampleDate: Date(timeIntervalSince1970: 100.5)
        ))
        XCTAssertFalse(FusionTimingPolicy.hasFreshImu(
            for: gnssDate,
            lastImuSampleDate: Date(timeIntervalSince1970: 98.9)
        ))
    }

    func testProcessingDateMaintainsMonotonicFilterTimestamps() {
        let previous = Date(timeIntervalSince1970: 10.0)
        let newer = Date(timeIntervalSince1970: 10.5)
        let older = Date(timeIntervalSince1970: 9.5)

        XCTAssertEqual(
            FusionTimingPolicy.processingDate(for: newer, after: previous),
            newer
        )
        XCTAssertEqual(
            FusionTimingPolicy.processingDate(for: older, after: previous),
            previous.addingTimeInterval(FusionTimingPolicy.timestampEpsilonSec)
        )
        XCTAssertEqual(
            FusionTimingPolicy.processingDate(for: previous, after: previous),
            previous.addingTimeInterval(FusionTimingPolicy.timestampEpsilonSec)
        )
    }
}
