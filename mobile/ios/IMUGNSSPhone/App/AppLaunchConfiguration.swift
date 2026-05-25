import Foundation

enum AppLaunchConfiguration {
    private static var arguments: [String] {
        ProcessInfo.processInfo.arguments
    }

    static var isUITesting: Bool {
        arguments.contains("-ui-testing")
    }

    static var shouldAutoStartSensors: Bool {
        !isUITesting
    }

    static var suppressesSensorHardware: Bool {
        isUITesting || arguments.contains("-suppress-sensor-hardware")
    }

    static func prepareProcessDefaults() {
        guard isUITesting else { return }

        let defaults = UserDefaults.standard
        defaults.removeObject(forKey: "developerToolsEnabled")
        MountMemoryDefaults.saveEnabled(false, to: defaults)
        MountMemoryDefaults.clearCalibration(from: defaults)
        HarshBehaviorPresetDefaults.save(.balanced, to: defaults)
        EventAudioSettingsDefaults.save(EventAudioSettings(), to: defaults)
    }
}
