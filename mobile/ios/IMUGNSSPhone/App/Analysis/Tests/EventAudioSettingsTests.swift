import XCTest
@testable import IMUGNSSPhone

final class EventAudioSettingsTests: XCTestCase {
    func testDefaultSettingsPlayVoiceAlertsInSilentMode() {
        let suiteName = "EventAudioSettingsTests.default.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let settings = EventAudioSettingsDefaults.load(from: defaults)

        XCTAssertEqual(settings.mode, .voice)
        XCTAssertTrue(settings.playDrivingAlertsInSilentMode)
    }

    func testSettingsPersistModeAndSilentModeBehavior() {
        let suiteName = "EventAudioSettingsTests.persist.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let expected = EventAudioSettings(mode: .chimeAndVoice, playDrivingAlertsInSilentMode: false)
        EventAudioSettingsDefaults.save(expected, to: defaults)

        XCTAssertEqual(EventAudioSettingsDefaults.load(from: defaults), expected)
    }

    func testAudioSessionPolicyDucksOtherAudioForDrivingAlerts() {
        let assertive = EventAudioPolicy.audioSessionConfiguration(
            for: EventAudioSettings(mode: .voice, playDrivingAlertsInSilentMode: true)
        )

        XCTAssertEqual(assertive.category, .playback)
        XCTAssertEqual(assertive.mode, .spokenAudio)
        XCTAssertTrue(assertive.options.contains(.mixWithOthers))
        XCTAssertTrue(assertive.options.contains(.duckOthers))
        XCTAssertTrue(assertive.options.contains(.interruptSpokenAudioAndMixWithOthers))

        let respectsSilentSwitch = EventAudioPolicy.audioSessionConfiguration(
            for: EventAudioSettings(mode: .voice, playDrivingAlertsInSilentMode: false)
        )

        XCTAssertEqual(respectsSilentSwitch.category, .ambient)
        XCTAssertEqual(respectsSilentSwitch.mode, .default)
        XCTAssertTrue(respectsSilentSwitch.options.contains(.mixWithOthers))
        XCTAssertTrue(respectsSilentSwitch.options.contains(.duckOthers))
    }

    func testVoiceModeUsesBackupChimeForAudibility() {
        XCTAssertFalse(EventAudioPolicy.shouldPlayChime(for: .off))
        XCTAssertTrue(EventAudioPolicy.shouldPlayChime(for: .chime))
        XCTAssertTrue(EventAudioPolicy.shouldPlayChime(for: .voice))
        XCTAssertTrue(EventAudioPolicy.shouldPlayChime(for: .chimeAndVoice))
    }

    func testForegroundAlertsDeliverWhenAudioIsEnabled() {
        XCTAssertTrue(EventAudioPolicy.shouldDeliverAlert(
            isForeground: true,
            settings: EventAudioSettings(mode: .voice, playDrivingAlertsInSilentMode: false)
        ))
        XCTAssertFalse(EventAudioPolicy.shouldDeliverAlert(
            isForeground: true,
            settings: EventAudioSettings(mode: .off, playDrivingAlertsInSilentMode: true)
        ))
    }

    func testBackgroundAlertsRequireDrivingAlertsAnyway() {
        XCTAssertTrue(EventAudioPolicy.shouldDeliverAlert(
            isForeground: false,
            settings: EventAudioSettings(mode: .voice, playDrivingAlertsInSilentMode: true)
        ))
        XCTAssertFalse(EventAudioPolicy.shouldDeliverAlert(
            isForeground: false,
            settings: EventAudioSettings(mode: .voice, playDrivingAlertsInSilentMode: false)
        ))
    }

    func testAlertCooldownUsesTimestampsInsteadOfRelativeEventSeconds() {
        let previous = Date(timeIntervalSinceReferenceDate: 10_000.0)

        XCTAssertTrue(EventAudioPolicy.shouldThrottleAlert(
            previousTimestamp: previous,
            nextTimestamp: previous.addingTimeInterval(1.0),
            minimumIntervalSec: 2.0
        ))
        XCTAssertFalse(EventAudioPolicy.shouldThrottleAlert(
            previousTimestamp: previous,
            nextTimestamp: previous.addingTimeInterval(2.1),
            minimumIntervalSec: 2.0
        ))
    }

    func testAlertCooldownSuppressesNonMonotonicTimestamps() {
        let previous = Date(timeIntervalSinceReferenceDate: 10_000.0)

        XCTAssertTrue(EventAudioPolicy.shouldThrottleAlert(
            previousTimestamp: previous,
            nextTimestamp: previous.addingTimeInterval(-9_900.0),
            minimumIntervalSec: 2.0
        ))
    }

    func testAudioPolicyDoesNotSpeakReadinessTransitions() {
        XCTAssertTrue(EventAudioPolicy.shouldPlayAudio(for: .harshBraking))
        XCTAssertTrue(EventAudioPolicy.shouldPlayAudio(for: .speedBump))
        XCTAssertTrue(EventAudioPolicy.shouldPlayAudio(for: .downhill))
        XCTAssertTrue(EventAudioPolicy.shouldPlayAudio(for: .uphill))
        XCTAssertFalse(EventAudioPolicy.shouldPlayAudio(for: .gnssDegraded))
        XCTAssertFalse(EventAudioPolicy.shouldPlayAudio(for: .mountReady))
        XCTAssertFalse(EventAudioPolicy.shouldPlayAudio(for: .fusionReady))
    }

    func testAudioAnnouncementsUseRequestedPhrases() {
        XCTAssertEqual(EventAudioPolicy.announcement(for: .harshAcceleration), "Harsh acceleration detected")
        XCTAssertEqual(EventAudioPolicy.announcement(for: .harshBraking), "Harsh braking detected")
        XCTAssertEqual(EventAudioPolicy.announcement(for: .harshCornering), "Harsh cornering detected")
        XCTAssertEqual(EventAudioPolicy.announcement(for: .reverse), "Reverse detected")
        XCTAssertEqual(EventAudioPolicy.announcement(for: .speedBump), "Speed bump detected")
        XCTAssertEqual(EventAudioPolicy.announcement(for: .downhill), "Downhill detected")
        XCTAssertEqual(EventAudioPolicy.announcement(for: .uphill), "Uphill detected")
    }

    func testChimeFactoryBuildsWavData() {
        let data = EventChimeFactory.wavData(frequencyHz: 440.0)

        XCTAssertGreaterThan(data.count, 44)
        XCTAssertEqual(String(decoding: data.prefix(4), as: UTF8.self), "RIFF")
        XCTAssertEqual(String(decoding: data.dropFirst(8).prefix(4), as: UTF8.self), "WAVE")
    }
}
