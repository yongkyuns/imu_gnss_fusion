import Foundation

enum RawSessionJSON {
    static func makeEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.nonConformingFloatEncodingStrategy = .convertToString(
            positiveInfinity: "Infinity",
            negativeInfinity: "-Infinity",
            nan: "NaN"
        )
        encoder.dateEncodingStrategy = .custom { date, encoder in
            var container = encoder.singleValueContainer()
            try container.encode(fractionalFormatter().string(from: date))
        }
        return encoder
    }

    static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.nonConformingFloatDecodingStrategy = .convertFromString(
            positiveInfinity: "Infinity",
            negativeInfinity: "-Infinity",
            nan: "NaN"
        )
        decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            let value = try container.decode(String.self)
            if let date = fractionalFormatter().date(from: value) ?? fallbackFormatter().date(from: value) {
                return date
            }
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Invalid ISO-8601 date: \(value)"
            )
        }
        return decoder
    }

    private static func fractionalFormatter() -> ISO8601DateFormatter {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }

    private static func fallbackFormatter() -> ISO8601DateFormatter {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }
}

struct RawSessionFileStore {
    let rootURL: URL
    private let readableRootURLs: [URL]

    init(rootURL: URL? = nil, legacyRootURLs: [URL]? = nil) {
        if let rootURL {
            self.rootURL = rootURL
            self.readableRootURLs = [rootURL] + (legacyRootURLs ?? [])
        } else {
            let baseURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
                ?? FileManager.default.temporaryDirectory
            self.rootURL = baseURL.appendingPathComponent("RawSessions", isDirectory: true)
            let legacyRootURL = FileManager.default
                .urls(for: .applicationSupportDirectory, in: .userDomainMask)
                .first?
                .appendingPathComponent("RawSessions", isDirectory: true)
            self.readableRootURLs = Self.uniqueURLs([self.rootURL, legacyRootURL].compactMap { $0 })
        }
    }

    func save(_ log: RawSessionLog) throws -> URL {
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        let url = rootURL.appendingPathComponent(Self.fileName(for: log))
        try RawSessionJSON.makeEncoder().encode(log).write(to: url, options: .atomic)
        try saveSummary(log.summary(fileURL: url), forSessionURL: url)
        return url
    }

    func writeSaveFailureDiagnostic(log: RawSessionLog, reason: String, error: Error) {
        try? FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        let url = rootURL.appendingPathComponent(Self.diagnosticFileName(for: log))
        let message = """
        reason: \(reason)
        error: \(String(reflecting: error))
        localized: \(error.localizedDescription)
        id: \(log.id.uuidString)
        name: \(log.name)
        startTime: \(RawSessionJSON.dateString(for: log.startTime))
        durationSec: \(log.durationSec)
        eventCount: \(log.events.count)
        imuCount: \(log.imuCount)
        gnssCount: \(log.gnssCount)
        barometerCount: \(log.barometerCount)
        """
        try? Data(message.utf8).write(to: url, options: .atomic)
    }

    func load(from url: URL) throws -> RawSessionLog {
        return try RawSessionJSON.makeDecoder().decode(RawSessionLog.self, from: Data(contentsOf: url))
    }

    func summaries() throws -> [RawSessionSummary] {
        var summariesByID: [UUID: RawSessionSummary] = [:]
        var orderedIDs: [UUID] = []
        for rootURL in readableRootURLs where FileManager.default.fileExists(atPath: rootURL.path) {
            let urls = try FileManager.default.contentsOfDirectory(
                at: rootURL,
                includingPropertiesForKeys: nil
            )
            for url in urls where url.pathExtension == "motionfusion" {
                if let summary = try loadSummary(forSessionURL: url) {
                    if summariesByID[summary.id] == nil {
                        orderedIDs.append(summary.id)
                    }
                    summariesByID[summary.id] = summary
                    continue
                }
                let summary = try load(from: url).summary(fileURL: url)
                try? saveSummary(summary, forSessionURL: url)
                if summariesByID[summary.id] == nil {
                    orderedIDs.append(summary.id)
                }
                summariesByID[summary.id] = summary
            }
        }
        return orderedIDs
            .compactMap { summariesByID[$0] }
            .sorted { lhs, rhs in
                lhs.startTime > rhs.startTime
            }
    }

    func delete(_ summary: RawSessionSummary) throws {
        guard let fileURL = summary.fileURL else { return }
        try FileManager.default.removeItem(at: fileURL)
        try? FileManager.default.removeItem(at: Self.summaryURL(forSessionURL: fileURL))
    }

    private func loadSummary(forSessionURL sessionURL: URL) throws -> RawSessionSummary? {
        let summaryURL = Self.summaryURL(forSessionURL: sessionURL)
        guard FileManager.default.fileExists(atPath: summaryURL.path) else { return nil }
        var summary = try RawSessionJSON.makeDecoder().decode(
            RawSessionSummary.self,
            from: Data(contentsOf: summaryURL)
        )
        summary.fileURL = sessionURL
        return summary
    }

    private func saveSummary(_ summary: RawSessionSummary, forSessionURL sessionURL: URL) throws {
        var persisted = summary
        persisted.fileURL = nil
        try RawSessionJSON.makeEncoder()
            .encode(persisted)
            .write(to: Self.summaryURL(forSessionURL: sessionURL), options: .atomic)
    }

    private static func summaryURL(forSessionURL sessionURL: URL) -> URL {
        sessionURL.deletingPathExtension().appendingPathExtension("summary.json")
    }

    private static func fileName(for log: RawSessionLog) -> String {
        "\(fileNamePrefix(for: log)).motionfusion"
    }

    private static func diagnosticFileName(for log: RawSessionLog) -> String {
        "\(fileNamePrefix(for: log)).save_error.txt"
    }

    private static func uniqueURLs(_ urls: [URL]) -> [URL] {
        var seen: Set<String> = []
        var unique: [URL] = []
        for url in urls {
            let key = url.standardizedFileURL.path
            guard !seen.contains(key) else { continue }
            seen.insert(key)
            unique.append(url)
        }
        return unique
    }

    private static func fileNamePrefix(for log: RawSessionLog) -> String {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd_HHmmss'Z'"
        let stamp = formatter.string(from: log.startTime)
        let shortID = log.id.uuidString.prefix(8)
        return "\(stamp)_\(shortID)"
    }
}

private extension RawSessionJSON {
    static func dateString(for date: Date) -> String {
        fractionalFormatter().string(from: date)
    }
}
