@preconcurrency import AVFoundation
import Foundation

final class EventAudioNotifier: NSObject, @unchecked Sendable {
    private let synthesizer = AVSpeechSynthesizer()
    private var activePlayers: [AVAudioPlayer] = []
    private var lastPlayedDateByKind: [MotionEvent.Kind: Date] = [:]
    private var lastSettings: EventAudioSettings?
    private let minimumIntervalSec = 2.0

    override init() {
        super.init()
        synthesizer.usesApplicationAudioSession = true
        synthesizer.delegate = self
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAudioSessionInterruption(_:)),
            name: AVAudioSession.interruptionNotification,
            object: AVAudioSession.sharedInstance()
        )
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    @MainActor
    func notify(_ event: MotionEvent, settings: EventAudioSettings) {
        guard settings.mode != .off else { return }
        guard EventAudioPolicy.shouldPlayAudio(for: event.kind) else { return }
        if EventAudioPolicy.shouldThrottleAlert(
            previousTimestamp: lastPlayedDateByKind[event.kind],
            nextTimestamp: event.timestamp,
            minimumIntervalSec: minimumIntervalSec
        ) {
            return
        }
        lastPlayedDateByKind[event.kind] = event.timestamp

        lastSettings = settings
        configureAudioSession(settings: settings)

        if EventAudioPolicy.shouldPlayChime(for: settings.mode) {
            playChime(for: event.kind)
        }
        if settings.mode.speaks {
            speak(EventAudioPolicy.announcement(for: event.kind))
        }
    }

    func reset() {
        DispatchQueue.main.async { [weak self] in
            self?.lastPlayedDateByKind.removeAll()
            self?.synthesizer.stopSpeaking(at: .immediate)
            self?.activePlayers.forEach { $0.stop() }
            self?.activePlayers.removeAll()
        }
    }

    @MainActor
    private func configureAudioSession(settings: EventAudioSettings) {
        let session = AVAudioSession.sharedInstance()
        let configuration = EventAudioPolicy.audioSessionConfiguration(for: settings)
        do {
            try session.setCategory(
                configuration.category,
                mode: configuration.mode,
                options: configuration.options
            )
            try session.setActive(true)
        } catch {
            #if DEBUG
            print("[EventAudioNotifier] Audio session configuration failed: \(error)")
            #endif
        }
    }

    @MainActor
    private func playChime(for kind: MotionEvent.Kind) {
        do {
            let player = try AVAudioPlayer(data: EventChimeFactory.wavData(frequencyHz: EventAudioPolicy.chimeFrequencyHz(for: kind)))
            player.delegate = self
            player.prepareToPlay()
            activePlayers.append(player)
            player.play()
            pruneInactivePlayers()
        } catch {
            #if DEBUG
            print("[EventAudioNotifier] Chime playback failed: \(error)")
            #endif
        }
    }

    @MainActor
    private func speak(_ text: String) {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        utterance.volume = 1.0
        utterance.preUtteranceDelay = 0.05
        utterance.postUtteranceDelay = 0.05
        synthesizer.speak(utterance)
    }

    @MainActor
    private func pruneInactivePlayers() {
        activePlayers.removeAll { !$0.isPlaying }
    }

    @objc
    private func handleAudioSessionInterruption(_ notification: Notification) {
        guard let rawType = notification.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: rawType),
              type == .ended else {
            return
        }
        Task { @MainActor [weak self] in
            guard let self, let lastSettings else { return }
            self.configureAudioSession(settings: lastSettings)
        }
    }
}

extension EventAudioNotifier: AVAudioPlayerDelegate {
    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor [weak self, weak player] in
            guard let player else { return }
            self?.activePlayers.removeAll { $0 === player }
        }
    }
}

extension EventAudioNotifier: AVSpeechSynthesizerDelegate {}

enum EventAudioPolicy {
    static func shouldThrottleAlert(
        previousTimestamp: Date?,
        nextTimestamp: Date,
        minimumIntervalSec: TimeInterval
    ) -> Bool {
        guard let previousTimestamp else { return false }
        return nextTimestamp.timeIntervalSince(previousTimestamp) < minimumIntervalSec
    }

    static func shouldDeliverAlert(isForeground: Bool, settings: EventAudioSettings) -> Bool {
        settings.mode != .off && (isForeground || settings.playDrivingAlertsInSilentMode)
    }

    static func audioSessionConfiguration(for settings: EventAudioSettings) -> EventAudioSessionConfiguration {
        if settings.playDrivingAlertsInSilentMode {
            return EventAudioSessionConfiguration(
                category: .playback,
                mode: .spokenAudio,
                options: [.mixWithOthers, .duckOthers, .interruptSpokenAudioAndMixWithOthers]
            )
        }
        return EventAudioSessionConfiguration(
            category: .ambient,
            mode: .default,
            options: [.mixWithOthers, .duckOthers]
        )
    }

    static func shouldPlayChime(for mode: EventAudibleAlertMode) -> Bool {
        switch mode {
        case .off:
            return false
        case .chime, .chimeAndVoice:
            return true
        case .voice:
            return true
        }
    }

    static func shouldPlayAudio(for kind: MotionEvent.Kind) -> Bool {
        switch kind {
        case .reverse, .harshAcceleration, .harshBraking, .harshCornering, .speedBump, .downhill, .uphill:
            return true
        case .gnssDegraded, .mountReady, .fusionReady:
            return false
        }
    }

    static func announcement(for kind: MotionEvent.Kind) -> String {
        switch kind {
        case .reverse: return "Reverse detected"
        case .harshAcceleration: return "Harsh acceleration detected"
        case .harshBraking: return "Harsh braking detected"
        case .harshCornering: return "Harsh cornering detected"
        case .speedBump: return "Speed bump detected"
        case .downhill: return "Downhill detected"
        case .uphill: return "Uphill detected"
        case .gnssDegraded: return "GNSS degraded"
        case .mountReady: return "Mount ready"
        case .fusionReady: return "Fusion ready"
        }
    }

    static func chimeFrequencyHz(for kind: MotionEvent.Kind) -> Double {
        switch kind {
        case .harshBraking: return 740.0
        case .harshCornering: return 660.0
        case .harshAcceleration: return 880.0
        case .reverse: return 520.0
        case .speedBump: return 784.0
        case .downhill: return 494.0
        case .uphill: return 622.0
        case .gnssDegraded: return 440.0
        case .mountReady, .fusionReady: return 587.0
        }
    }
}

enum EventChimeFactory {
    static func wavData(frequencyHz: Double, durationSec: Double = 0.16, sampleRate: Int = 44_100) -> Data {
        let sampleCount = max(1, Int(Double(sampleRate) * durationSec))
        var pcm = Data(capacity: sampleCount * MemoryLayout<Int16>.size)
        for index in 0..<sampleCount {
            let t = Double(index) / Double(sampleRate)
            let envelope = min(1.0, Double(index) / 800.0) * min(1.0, Double(sampleCount - index) / 1_200.0)
            let sample = sin(2.0 * .pi * frequencyHz * t) * 0.35 * envelope
            var intSample = Int16(sample * Double(Int16.max)).littleEndian
            withUnsafeBytes(of: &intSample) { pcm.append(contentsOf: $0) }
        }
        return wavHeader(pcmByteCount: pcm.count, sampleRate: sampleRate) + pcm
    }

    private static func wavHeader(pcmByteCount: Int, sampleRate: Int) -> Data {
        var data = Data()
        data.append(contentsOf: "RIFF".utf8)
        data.append(littleEndianUInt32(UInt32(36 + pcmByteCount)))
        data.append(contentsOf: "WAVE".utf8)
        data.append(contentsOf: "fmt ".utf8)
        data.append(littleEndianUInt32(16))
        data.append(littleEndianUInt16(1))
        data.append(littleEndianUInt16(1))
        data.append(littleEndianUInt32(UInt32(sampleRate)))
        data.append(littleEndianUInt32(UInt32(sampleRate * 2)))
        data.append(littleEndianUInt16(2))
        data.append(littleEndianUInt16(16))
        data.append(contentsOf: "data".utf8)
        data.append(littleEndianUInt32(UInt32(pcmByteCount)))
        return data
    }

    private static func littleEndianUInt16(_ value: UInt16) -> Data {
        var littleEndian = value.littleEndian
        return Data(bytes: &littleEndian, count: MemoryLayout<UInt16>.size)
    }

    private static func littleEndianUInt32(_ value: UInt32) -> Data {
        var littleEndian = value.littleEndian
        return Data(bytes: &littleEndian, count: MemoryLayout<UInt32>.size)
    }
}
