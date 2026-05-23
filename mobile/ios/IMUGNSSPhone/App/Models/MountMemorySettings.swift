import Foundation

struct SavedMountCalibration: Equatable, Sendable {
    var qBV: Quaternion
    var savedAt: Date

    init?(qBV: Quaternion, savedAt: Date) {
        guard let normalized = MountMemoryPolicy.validNormalizedQuaternion(qBV) else {
            return nil
        }
        self.qBV = normalized
        self.savedAt = savedAt
    }
}

struct MountMemorySettings: Equatable, Sendable {
    var isEnabled: Bool = false
    var savedCalibration: SavedMountCalibration?

    var canUseSavedMount: Bool {
        isEnabled && savedCalibration != nil
    }
}

enum MountMemoryPolicy {
    static let minimumStoreAngularDeltaDeg = 0.25

    static func validNormalizedQuaternion(_ q: Quaternion) -> Quaternion? {
        let norm = q.norm
        guard norm.isFinite, norm > 1e-6 else { return nil }
        let normalized = q.normalized
        guard normalized.w.isFinite,
              normalized.x.isFinite,
              normalized.y.isFinite,
              normalized.z.isFinite else {
            return nil
        }
        return normalized
    }

    static func shouldStore(previous: SavedMountCalibration?, next qBV: Quaternion) -> Bool {
        guard let next = validNormalizedQuaternion(qBV) else { return false }
        guard let previous else { return true }
        return angularDistanceDeg(previous.qBV, next) >= minimumStoreAngularDeltaDeg
    }

    static func activeModeTitle(_ settings: MountMemorySettings) -> String {
        guard settings.isEnabled else { return "Auto Align" }
        return settings.savedCalibration == nil ? "Auto Align, Saving Mount" : "Remembered Mount"
    }

    static func savedMountTitle(_ settings: MountMemorySettings) -> String {
        guard let savedAt = settings.savedCalibration?.savedAt else { return "None" }
        return savedAt.formatted(date: .abbreviated, time: .shortened)
    }

    private static func angularDistanceDeg(_ lhs: Quaternion, _ rhs: Quaternion) -> Double {
        let a = lhs.normalized
        let b = rhs.normalized
        let dot = min(1.0, max(-1.0, abs(a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z)))
        return 2.0 * acos(dot) * 180.0 / .pi
    }
}

enum MountMemoryDefaults {
    private static let enabledKey = "mountMemoryEnabled"
    private static let savedAtKey = "mountMemorySavedAt"
    private static let qWKey = "mountMemoryQBVW"
    private static let qXKey = "mountMemoryQBVX"
    private static let qYKey = "mountMemoryQBVY"
    private static let qZKey = "mountMemoryQBVZ"

    static func load(from defaults: UserDefaults = .standard) -> MountMemorySettings {
        let isEnabled = defaults.bool(forKey: enabledKey)
        let savedCalibration = loadCalibration(from: defaults)
        return MountMemorySettings(isEnabled: isEnabled, savedCalibration: savedCalibration)
    }

    static func saveEnabled(_ isEnabled: Bool, to defaults: UserDefaults = .standard) {
        defaults.set(isEnabled, forKey: enabledKey)
    }

    static func saveCalibration(_ calibration: SavedMountCalibration, to defaults: UserDefaults = .standard) {
        let q = calibration.qBV.normalized
        defaults.set(q.w, forKey: qWKey)
        defaults.set(q.x, forKey: qXKey)
        defaults.set(q.y, forKey: qYKey)
        defaults.set(q.z, forKey: qZKey)
        defaults.set(calibration.savedAt.timeIntervalSince1970, forKey: savedAtKey)
    }

    static func clearCalibration(from defaults: UserDefaults = .standard) {
        [savedAtKey, qWKey, qXKey, qYKey, qZKey].forEach { defaults.removeObject(forKey: $0) }
    }

    private static func loadCalibration(from defaults: UserDefaults) -> SavedMountCalibration? {
        guard defaults.object(forKey: qWKey) != nil,
              defaults.object(forKey: qXKey) != nil,
              defaults.object(forKey: qYKey) != nil,
              defaults.object(forKey: qZKey) != nil else {
            return nil
        }
        let savedAtSeconds = defaults.object(forKey: savedAtKey) == nil
            ? Date().timeIntervalSince1970
            : defaults.double(forKey: savedAtKey)
        guard savedAtSeconds.isFinite else { return nil }
        return SavedMountCalibration(
            qBV: Quaternion(
                w: defaults.double(forKey: qWKey),
                x: defaults.double(forKey: qXKey),
                y: defaults.double(forKey: qYKey),
                z: defaults.double(forKey: qZKey)
            ),
            savedAt: Date(timeIntervalSince1970: savedAtSeconds)
        )
    }
}
