import Foundation

enum EventAudibleAlertMode: String, CaseIterable, Identifiable, Sendable {
    case off
    case chime
    case voice
    case chimeAndVoice

    var id: String { rawValue }

    var displayTitle: String {
        switch self {
        case .off: return "Off"
        case .chime: return "Chime"
        case .voice: return "Voice"
        case .chimeAndVoice: return "Chime + Voice"
        }
    }

    var playsChime: Bool {
        self == .chime || self == .chimeAndVoice
    }

    var speaks: Bool {
        self == .voice || self == .chimeAndVoice
    }
}

struct EventAudioSettings: Equatable, Sendable {
    var mode: EventAudibleAlertMode = .voice
    var playDrivingAlertsInSilentMode: Bool = true
}

enum EventAudioSettingsDefaults {
    private static let modeKey = "eventAudibleAlertMode"
    private static let playInSilentModeKey = "eventPlayDrivingAlertsInSilentMode"

    static func load(from defaults: UserDefaults = .standard) -> EventAudioSettings {
        let mode = defaults.string(forKey: modeKey)
            .flatMap(EventAudibleAlertMode.init(rawValue:)) ?? EventAudioSettings().mode
        let playInSilentMode: Bool
        if defaults.object(forKey: playInSilentModeKey) == nil {
            playInSilentMode = EventAudioSettings().playDrivingAlertsInSilentMode
        } else {
            playInSilentMode = defaults.bool(forKey: playInSilentModeKey)
        }
        return EventAudioSettings(mode: mode, playDrivingAlertsInSilentMode: playInSilentMode)
    }

    static func save(_ settings: EventAudioSettings, to defaults: UserDefaults = .standard) {
        defaults.set(settings.mode.rawValue, forKey: modeKey)
        defaults.set(settings.playDrivingAlertsInSilentMode, forKey: playInSilentModeKey)
    }
}
