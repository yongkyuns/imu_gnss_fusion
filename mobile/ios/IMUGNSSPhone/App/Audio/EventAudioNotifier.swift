@preconcurrency import AVFoundation
import Foundation

final class EventAudioNotifier: NSObject, @unchecked Sendable {
    private let synthesizer = AVSpeechSynthesizer()
    private var activePlayers: [AVAudioPlayer] = []
    private var lastPlayedTSecByKind: [MotionEvent.Kind: Double] = [:]
    private let minimumIntervalSec = 2.0

    override init() {
        super.init()
        synthesizer.usesApplicationAudioSession = true
        synthesizer.delegate = self
    }

    @MainActor
    func notify(_ event: MotionEvent, settings: EventAudioSettings) {
        guard settings.mode != .off else { return }
        guard EventAudioPolicy.shouldPlayAudio(for: event.kind) else { return }
        if let lastTSec = lastPlayedTSecByKind[event.kind],
           event.tSec - lastTSec < minimumIntervalSec {
            return
        }
        lastPlayedTSecByKind[event.kind] = event.tSec

        configureAudioSession(settings: settings)

        if settings.mode.playsChime {
            playChime(for: event.kind)
        }
        if settings.mode.speaks {
            speak(EventAudioPolicy.announcement(for: event.kind))
        }
    }

    func reset() {
        DispatchQueue.main.async { [weak self] in
            self?.lastPlayedTSecByKind.removeAll()
            self?.synthesizer.stopSpeaking(at: .immediate)
            self?.activePlayers.forEach { $0.stop() }
            self?.activePlayers.removeAll()
        }
    }

    @MainActor
    private func configureAudioSession(settings: EventAudioSettings) {
        let session = AVAudioSession.sharedInstance()
        do {
            if settings.playDrivingAlertsInSilentMode {
                try session.setCategory(.playback, mode: .spokenAudio, options: [.mixWithOthers])
            } else {
                try session.setCategory(.ambient, mode: .default, options: [.mixWithOthers])
            }
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
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 0.92
        utterance.volume = 1.0
        utterance.preUtteranceDelay = 0.05
        utterance.postUtteranceDelay = 0.05
        synthesizer.speak(utterance)
    }

    @MainActor
    private func pruneInactivePlayers() {
        activePlayers.removeAll { !$0.isPlaying }
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
