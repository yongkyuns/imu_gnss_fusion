import XCTest
@testable import IMUGNSSPhone

final class MountMemorySettingsTests: XCTestCase {
    func testMountMemoryDefaultsRoundTripEnabledAndCalibration() {
        let suiteName = "MountMemorySettingsTests.roundTrip.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let savedAt = Date(timeIntervalSince1970: 1_764_000_000)
        let calibration = SavedMountCalibration(
            qBV: Quaternion(w: 2.0, x: 0.0, y: 0.0, z: 0.0),
            savedAt: savedAt
        )!

        MountMemoryDefaults.saveEnabled(true, to: defaults)
        MountMemoryDefaults.saveCalibration(calibration, to: defaults)

        let loaded = MountMemoryDefaults.load(from: defaults)
        XCTAssertTrue(loaded.isEnabled)
        XCTAssertEqual(loaded.savedCalibration?.qBV, .identity)
        XCTAssertEqual(loaded.savedCalibration?.savedAt.timeIntervalSince1970 ?? 0.0, savedAt.timeIntervalSince1970)
        XCTAssertTrue(loaded.canUseSavedMount)
    }

    func testMountMemoryRejectsInvalidQuaternion() {
        XCTAssertNil(SavedMountCalibration(
            qBV: Quaternion(w: .nan, x: 0.0, y: 0.0, z: 0.0),
            savedAt: Date()
        ))
        XCTAssertNil(SavedMountCalibration(
            qBV: Quaternion(w: 0.0, x: 0.0, y: 0.0, z: 0.0),
            savedAt: Date()
        ))
    }

    func testMountMemoryStorePolicyIgnoresTinyChanges() {
        let previous = SavedMountCalibration(
            qBV: Quaternion(w: 1.0, x: 0.0, y: 0.0, z: 0.0),
            savedAt: Date()
        )!
        let tinyYaw = Quaternion(
            w: cos(0.001 * .pi / 180.0 / 2.0),
            x: 0.0,
            y: 0.0,
            z: sin(0.001 * .pi / 180.0 / 2.0)
        )
        let largerYaw = Quaternion(
            w: cos(1.0 * .pi / 180.0 / 2.0),
            x: 0.0,
            y: 0.0,
            z: sin(1.0 * .pi / 180.0 / 2.0)
        )

        XCTAssertFalse(MountMemoryPolicy.shouldStore(previous: previous, next: tinyYaw))
        XCTAssertTrue(MountMemoryPolicy.shouldStore(previous: previous, next: largerYaw))
    }

    func testMountMemoryTitlesDescribeFallbackAndSavedMode() {
        XCTAssertEqual(MountMemoryPolicy.activeModeTitle(MountMemorySettings()), "Auto Align")
        XCTAssertEqual(
            MountMemoryPolicy.activeModeTitle(MountMemorySettings(isEnabled: true, savedCalibration: nil)),
            "Auto Align, Saving Mount"
        )

        let saved = SavedMountCalibration(qBV: .identity, savedAt: Date())!
        XCTAssertEqual(
            MountMemoryPolicy.activeModeTitle(MountMemorySettings(isEnabled: true, savedCalibration: saved)),
            "Remembered Mount"
        )
    }
}
