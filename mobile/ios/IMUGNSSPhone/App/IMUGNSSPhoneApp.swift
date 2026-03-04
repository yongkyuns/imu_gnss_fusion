import SwiftUI

@main
struct IMUGNSSPhoneApp: App {
    @StateObject private var store = SensorStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(store)
                .onAppear {
                    store.start()
                }
        }
    }
}
