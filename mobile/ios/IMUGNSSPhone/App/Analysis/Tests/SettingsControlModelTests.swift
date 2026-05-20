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
}
