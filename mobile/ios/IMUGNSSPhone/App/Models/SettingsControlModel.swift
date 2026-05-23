import Combine
import CoreLocation
import Foundation

struct SettingsControlState: Equatable {
    var authorization: CLAuthorizationStatus = .notDetermined
    var streamMode: SensorStore.StreamMode = .live
    var isLiveSensorStreamRunning: Bool = false
    var activeSessionName: String?
    var replayProgress: Double = 0.0
    var playbackSpeedMultiplier: Double = PlaybackSpeedPolicy.defaultMultiplier
    var eventAudioSettings: EventAudioSettings = EventAudioSettings()
    var mountMemorySettings: MountMemorySettings = MountMemorySettings()
    var isRecording: Bool = false
    var savedSessionCount: Int = 0

    var streamStatusTitle: String {
        switch streamMode {
        case .playback:
            return streamMode.rawValue
        case .live:
            return isLiveSensorStreamRunning ? streamMode.rawValue : "Stopped"
        }
    }

    var playbackSpeedTitle: String {
        PlaybackSpeedPolicy.title(for: playbackSpeedMultiplier)
    }

    var mountModeTitle: String {
        MountMemoryPolicy.activeModeTitle(mountMemorySettings)
    }

    var savedMountTitle: String {
        MountMemoryPolicy.savedMountTitle(mountMemorySettings)
    }
}

enum PlaybackSpeedPolicy {
    static let defaultMultiplier = 10.0
    static let minimumMultiplier = 0.1
    static let maximumMultiplier = 50.0
    static let options = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    static func normalized(_ multiplier: Double) -> Double {
        guard multiplier.isFinite else { return defaultMultiplier }
        return min(max(multiplier, minimumMultiplier), maximumMultiplier)
    }

    static func title(for multiplier: Double) -> String {
        let normalized = normalized(multiplier)
        if normalized.rounded() == normalized {
            return "\(Int(normalized))x"
        }
        return String(format: "%.1fx", normalized)
    }
}

final class SettingsControlModel: ObservableObject {
    @Published private(set) var state = SettingsControlState()

    private weak var sensorStore: SensorStore?

    func bind(sensorStore: SensorStore) {
        self.sensorStore = sensorStore
    }

    func update(_ nextState: SettingsControlState) {
        guard nextState != state else { return }
        state = nextState
    }

    func startSensors() {
        sensorStore?.start()
    }

    func stopSensors() {
        sensorStore?.stop()
    }

    func stopPlayback() {
        sensorStore?.stopPlayback()
    }

    func setPlaybackSpeedMultiplier(_ multiplier: Double) {
        sensorStore?.setPlaybackSpeedMultiplier(multiplier)
    }

    func setEventAudibleAlertMode(_ mode: EventAudibleAlertMode) {
        sensorStore?.setEventAudibleAlertMode(mode)
    }

    func setEventAlertsPlayInSilentMode(_ isEnabled: Bool) {
        sensorStore?.setEventAlertsPlayInSilentMode(isEnabled)
    }

    func playTestEventAudioAlert() {
        sensorStore?.playTestEventAudioAlert()
    }

    func setMountMemoryEnabled(_ isEnabled: Bool) {
        sensorStore?.setMountMemoryEnabled(isEnabled)
    }

    func clearRememberedMount() {
        sensorStore?.clearRememberedMount()
    }

    func startRecording() {
        sensorStore?.startRecording()
    }

    func stopRecording() {
        sensorStore?.stopRecording()
    }

    func refreshSessions() {
        sensorStore?.loadRecordedSessions()
    }
}
