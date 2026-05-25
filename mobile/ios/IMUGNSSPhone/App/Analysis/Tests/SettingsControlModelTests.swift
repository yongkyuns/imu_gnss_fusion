import Combine
import XCTest
@testable import IMUGNSSPhone

final class SettingsControlModelTests: XCTestCase {
    func testUpdatePublishesOnlyWhenSettingsStateChanges() {
        let model = SettingsControlModel()
        var publishCount = 0
        let cancellable = model.objectWillChange.sink {
            publishCount += 1
        }

        model.update(SettingsControlState())
        XCTAssertEqual(publishCount, 0)

        model.update(SettingsControlState(savedSessionCount: 1))
        XCTAssertEqual(publishCount, 1)

        model.update(SettingsControlState(savedSessionCount: 1))
        XCTAssertEqual(publishCount, 1)

        cancellable.cancel()
    }

    func testStreamStatusDistinguishesLiveStoppedAndPlayback() {
        XCTAssertEqual(
            SettingsControlState(streamMode: .live, isLiveSensorStreamRunning: true).streamStatusTitle,
            "Live"
        )
        XCTAssertEqual(
            SettingsControlState(streamMode: .live, isLiveSensorStreamRunning: false).streamStatusTitle,
            "Stopped"
        )
        XCTAssertEqual(
            SettingsControlState(streamMode: .playback, isLiveSensorStreamRunning: false).streamStatusTitle,
            "Playback"
        )
    }

    func testPlaybackSpeedPolicyNormalizesAndFormatsSpeed() {
        XCTAssertEqual(PlaybackSpeedPolicy.normalized(.nan), PlaybackSpeedPolicy.defaultMultiplier)
        XCTAssertEqual(PlaybackSpeedPolicy.normalized(0.01), PlaybackSpeedPolicy.minimumMultiplier)
        XCTAssertEqual(PlaybackSpeedPolicy.normalized(100.0), PlaybackSpeedPolicy.maximumMultiplier)
        XCTAssertEqual(PlaybackSpeedPolicy.title(for: 10.0), "10x")
        XCTAssertEqual(PlaybackSpeedPolicy.title(for: 0.5), "0.5x")
    }

    func testSettingsStateExposesPlaybackSpeedTitle() {
        let state = SettingsControlState(playbackSpeedMultiplier: 2.0)
        XCTAssertEqual(state.playbackSpeedTitle, "2x")
    }

    func testSettingsStateCarriesHarshBehaviorPreset() {
        let state = SettingsControlState(harshBehaviorPreset: .conservative)

        XCTAssertEqual(state.harshBehaviorPreset, .conservative)
        XCTAssertEqual(state.harshBehaviorTitle, "Conservative")
    }

    func testSettingsStateCarriesEventAudioSettings() {
        let settings = EventAudioSettings(mode: .chimeAndVoice, playDrivingAlertsInSilentMode: true)
        let state = SettingsControlState(eventAudioSettings: settings)

        XCTAssertEqual(state.eventAudioSettings, settings)
    }

    func testSettingsStateExposesMountMemoryTitles() {
        let saved = SavedMountCalibration(qBV: .identity, savedAt: Date())!
        let state = SettingsControlState(
            mountMemorySettings: MountMemorySettings(isEnabled: true, savedCalibration: saved)
        )

        XCTAssertEqual(state.mountModeTitle, "Remembered Mount")
        XCTAssertNotEqual(state.savedMountTitle, "None")
    }

    func testEventAudioModeTitlesAreStable() {
        XCTAssertEqual(EventAudibleAlertMode.off.displayTitle, "Off")
        XCTAssertEqual(EventAudibleAlertMode.chime.displayTitle, "Chime")
        XCTAssertEqual(EventAudibleAlertMode.voice.displayTitle, "Voice")
        XCTAssertEqual(EventAudibleAlertMode.chimeAndVoice.displayTitle, "Chime + Voice")
    }

    func testHarshBehaviorPresetDefaultsRoundTrip() {
        let suiteName = "HarshBehaviorPresetDefaults.roundTrip.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }

        XCTAssertEqual(HarshBehaviorPresetDefaults.load(from: defaults), .balanced)
        HarshBehaviorPresetDefaults.save(.sensitive, to: defaults)
        XCTAssertEqual(HarshBehaviorPresetDefaults.load(from: defaults), .sensitive)
        defaults.set(999, forKey: "harshBehaviorPreset")
        XCTAssertEqual(HarshBehaviorPresetDefaults.load(from: defaults), .balanced)
    }
}
