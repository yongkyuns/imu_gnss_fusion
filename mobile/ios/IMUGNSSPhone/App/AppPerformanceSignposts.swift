import Foundation
import os

enum AppPerformanceSignposts {
    private static let signposter = OSSignposter(
        subsystem: Bundle.main.bundleIdentifier ?? "com.ykshin.ekf.imugnssphone",
        category: "Performance"
    )

    static func interval<T>(_ name: StaticString, _ work: () -> T) -> T {
        let state = signposter.beginInterval(name)
        defer { signposter.endInterval(name, state) }
        return work()
    }

    static func event(_ name: StaticString) {
        signposter.emitEvent(name)
    }
}
