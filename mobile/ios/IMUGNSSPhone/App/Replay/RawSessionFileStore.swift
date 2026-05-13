import Foundation

enum RawSessionJSON {
    static func makeEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .custom { date, encoder in
            var container = encoder.singleValueContainer()
            try container.encode(fractionalFormatter().string(from: date))
        }
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }

    static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
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

    init(rootURL: URL? = nil) {
        if let rootURL {
            self.rootURL = rootURL
        } else {
            let baseURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
                ?? FileManager.default.temporaryDirectory
            self.rootURL = baseURL.appendingPathComponent("RawSessions", isDirectory: true)
        }
    }

    func save(_ log: RawSessionLog) throws -> URL {
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        let url = rootURL.appendingPathComponent(Self.fileName(for: log))
        try RawSessionJSON.makeEncoder().encode(log).write(to: url, options: .atomic)
        return url
    }

    func load(from url: URL) throws -> RawSessionLog {
        return try RawSessionJSON.makeDecoder().decode(RawSessionLog.self, from: Data(contentsOf: url))
    }

    func summaries() throws -> [RawSessionSummary] {
        guard FileManager.default.fileExists(atPath: rootURL.path) else { return [] }
        let urls = try FileManager.default.contentsOfDirectory(
            at: rootURL,
            includingPropertiesForKeys: nil
        )
        return try urls
            .filter { $0.pathExtension == "motionfusion" }
            .map { url in
                try load(from: url).summary(fileURL: url)
            }
            .sorted { lhs, rhs in
                lhs.startTime > rhs.startTime
            }
    }

    func delete(_ summary: RawSessionSummary) throws {
        guard let fileURL = summary.fileURL else { return }
        try FileManager.default.removeItem(at: fileURL)
    }

    private static func fileName(for log: RawSessionLog) -> String {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd_HHmmss'Z'"
        let stamp = formatter.string(from: log.startTime)
        let shortID = log.id.uuidString.prefix(8)
        return "\(stamp)_\(shortID).motionfusion"
    }
}
