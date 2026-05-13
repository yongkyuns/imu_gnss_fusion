import SwiftUI
import UIKit

@main
struct IMUGNSSPhoneApp: App {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var store = SensorStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(store)
                .onAppear {
                    UIApplication.shared.isIdleTimerDisabled = true
                    store.start()
                }
                .onChange(of: scenePhase) { phase in
                    switch phase {
                    case .active:
                        UIApplication.shared.isIdleTimerDisabled = true
                        store.prepareForForegroundTracking()
                    case .inactive:
                        UIApplication.shared.isIdleTimerDisabled = false
                        store.checkpointRecording()
                    case .background:
                        UIApplication.shared.isIdleTimerDisabled = false
                        store.prepareForBackgroundTracking()
                    @unknown default:
                        UIApplication.shared.isIdleTimerDisabled = false
                        store.checkpointRecording()
                    }
                }
                .onReceive(NotificationCenter.default.publisher(for: UIApplication.willTerminateNotification)) { _ in
                    UIApplication.shared.isIdleTimerDisabled = false
                    store.applicationWillTerminate()
                }
        }
    }
}
