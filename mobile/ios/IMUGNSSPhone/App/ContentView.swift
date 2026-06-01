import Charts
import CoreLocation
import MapKit
import SwiftUI
import UIKit

private enum ChartAxisMode: String, CaseIterable, Identifiable {
    case x = "X"
    case y = "Y"
    case xy = "XY"

    var id: String { rawValue }

    var allowsX: Bool { self == .x || self == .xy }
    var allowsY: Bool { self == .y || self == .xy }
}

private enum ChartKind: String, CaseIterable, Identifiable {
    case nedPosition
    case nedVelocity
    case imuAccel
    case imuGyro
    case ekfVelocity
    case ekfEuler
    case ekfGyroBias
    case ekfAccelBias

    var id: String { rawValue }

    var title: String {
        switch self {
        case .nedPosition: return "NED Position (m)"
        case .nedVelocity: return "NED Velocity (m/s)"
        case .imuAccel: return "Accel (m/s²)"
        case .imuGyro: return "Gyro (rad/s)"
        case .ekfVelocity: return "EKF Velocity (m/s)"
        case .ekfEuler: return "EKF Euler (deg)"
        case .ekfGyroBias: return "EKF Gyro Bias (rad/s)"
        case .ekfAccelBias: return "EKF Accel Bias (m/s²)"
        }
    }

    var axisLabels: (String, String, String) {
        switch self {
        case .nedPosition: return ("N", "E", "D")
        case .nedVelocity: return ("Vn", "Ve", "Vd")
        case .imuAccel: return ("Ax", "Ay", "Az")
        case .imuGyro: return ("Gx", "Gy", "Gz")
        case .ekfVelocity: return ("Vn", "Ve", "Vd")
        case .ekfEuler: return ("Roll", "Pitch", "Yaw")
        case .ekfGyroBias: return ("bgx", "bgy", "bgz")
        case .ekfAccelBias: return ("bax", "bay", "baz")
        }
    }
}

private enum DrivePanelDetent: CaseIterable {
    case collapsed
    case medium
    case expanded

    var systemImage: String {
        switch self {
        case .collapsed: return "chevron.up"
        case .medium: return "chevron.up.chevron.down"
        case .expanded: return "chevron.down"
        }
    }

    func next() -> DrivePanelDetent {
        switch self {
        case .collapsed: return .medium
        case .medium: return .expanded
        case .expanded: return .collapsed
        }
    }

    var followCameraVerticalOffset: CGFloat {
        switch self {
        case .collapsed: return 0.0
        case .medium: return 96.0
        case .expanded: return 168.0
        }
    }
}

private enum DriveTelemetryTab: String, CaseIterable, Identifiable {
    case metrics = "Metrics"
    case streams = "Streams"

    var id: String { rawValue }
}

private extension RouteLayerSelection {
    var displayTitle: String {
        switch self {
        case .none: return "None"
        case .fused: return "Fused"
        case .gnss: return "GNSS"
        case .both: return "Both"
        case .delta: return "Compare"
        }
    }

    var systemImage: String {
        switch self {
        case .none: return "eye.slash"
        case .fused: return "location.north.line.fill"
        case .gnss: return "location.fill"
        case .both: return "map.fill"
        case .delta: return "point.topleft.down.curvedto.point.bottomright.up"
        }
    }
}

struct ContentView: View {
#if DEBUG
    @State private var developerToolsEnabled = UserDefaults.standard.bool(forKey: "developerToolsEnabled")
#endif

    var body: some View {
        TabView {
            DriveView()
                .tabItem {
                    Label("Drive", systemImage: "map")
                }

            ReviewView()
                .tabItem {
                    Label("Review", systemImage: "clock.arrow.circlepath")
                }

#if DEBUG
            SettingsView(developerToolsEnabled: $developerToolsEnabled)
                .tabItem {
                    Label("Settings", systemImage: "gearshape")
                }
#else
            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gearshape")
                }
#endif

#if DEBUG
            if developerToolsEnabled {
                DiagnosticsView()
                    .tabItem {
                        Label("Diagnostics", systemImage: "waveform.path.ecg")
                    }
            }
#endif
        }
#if DEBUG
        .id(developerToolsEnabled)
        .onChange(of: developerToolsEnabled) { isEnabled in
            UserDefaults.standard.set(isEnabled, forKey: "developerToolsEnabled")
        }
#endif
    }
}

private struct DriveView: View {
    @EnvironmentObject private var store: SensorStore
    @State private var panelDetent: DrivePanelDetent = .collapsed
    @State private var routeLayer: RouteLayerSelection = .both
    @State private var showsAccuracyOverlay = false
    @State private var viewportRefreshToken = 0
    @State private var followsCurrentMarker = true
    @State private var headsUpMotionEvent: MotionEvent?

    private var rawRouteCoordinates: [CLLocationCoordinate2D] {
        guard routeLayer.showsGnssRoute else { return [] }
        return RawGNSSRoute.coordinates(
            currentLatitude: store.latitude,
            currentLongitude: store.longitude,
            currentNorthM: store.posNorthM,
            currentEastM: store.posEastM,
            positionHistory: store.gnssRouteHistory
        )
    }

    private var fusedRouteCoordinates: [CLLocationCoordinate2D] {
        guard routeLayer.showsFusedRoute else { return [] }
        guard showsFusedOutput else { return [] }
        return MapGeographicRoutePolicy.coordinates(
            history: store.fusedRouteCoordinateHistory.map(\.coordinate),
            current: currentFusedGeographicCoordinate
        )
    }

    private var currentCoordinate: CLLocationCoordinate2D? {
        guard routeLayer.showsGnssRoute else { return nil }
        guard let latitude = store.latitude, let longitude = store.longitude else { return nil }
        return CLLocationCoordinate2D(latitude: latitude, longitude: longitude)
    }

    private var fusedCurrentCoordinate: CLLocationCoordinate2D? {
        guard routeLayer.showsFusedRoute else { return nil }
        guard showsFusedOutput else { return nil }
        guard let latitude = store.fusedLatitude, let longitude = store.fusedLongitude else { return nil }
        return CLLocationCoordinate2D(latitude: latitude, longitude: longitude)
    }

    private var currentFusedGeographicCoordinate: GeographicCoordinate? {
        guard let latitude = store.fusedLatitude, let longitude = store.fusedLongitude else { return nil }
        return GeographicCoordinate(
            latitudeDeg: latitude,
            longitudeDeg: longitude,
            altitudeM: store.fusedAltitudeM
        )
    }

    private var mapEvents: [MotionEvent] {
        Array(store.motionEvents.suffix(60))
    }

    private var currentMapHeadingDeg: Double? {
        if showsFusedOutput,
           let yawDeg = store.ekfEulerHistory.last?.z,
           yawDeg.isFinite {
            return MapFollowPolicy.normalizedHeadingDeg(yawDeg)
        }
        guard let courseDeg = store.courseDeg, courseDeg.isFinite else { return nil }
        return MapFollowPolicy.normalizedHeadingDeg(courseDeg)
    }

    private var showsFusedOutput: Bool {
        FusedMapVisibilityPolicy.shouldShowFusedOutput(
            initialized: store.ekfInitialized,
            mountReady: store.ekfMountReady
        )
    }

    private var gnssQuality: GNSSQuality {
        GNSSQuality(horizontalAccuracyM: store.horizontalAccuracyM, timestamp: store.locationTimestamp)
    }

    var body: some View {
        NavigationStack {
            ZStack {
                RawGNSSMapView(
                    gnssCoordinates: rawRouteCoordinates,
                    fusedCoordinates: fusedRouteCoordinates,
                    currentCoordinate: currentCoordinate,
                    fusedCurrentCoordinate: fusedCurrentCoordinate,
                    eventAnnotations: mapEvents,
                    currentHeadingDeg: currentMapHeadingDeg,
                    horizontalAccuracyM: store.horizontalAccuracyM,
                    showAccuracyOverlay: showsAccuracyOverlay,
                    followsCurrentMarker: followsCurrentMarker,
                    followCameraVerticalOffset: panelDetent.followCameraVerticalOffset,
                    viewportRefreshToken: viewportRefreshToken
                )
                .ignoresSafeArea(edges: .top)

                VStack(spacing: 0) {
                    TopMapStatusBar(gnssQuality: gnssQuality)
                        .environmentObject(store)
                        .padding(.top, 8)
                        .padding(.horizontal, 12)

                    if let event = headsUpMotionEvent {
                        MotionEventHeadsUp(event: event)
                            .padding(.top, 8)
                            .padding(.horizontal, 12)
                            .transition(.move(edge: .top).combined(with: .opacity))
                    }

                    Spacer()

                    PlaybackMapControlPanel()
                        .environmentObject(store)
                        .padding(.horizontal, 12)
                        .padding(.bottom, store.streamMode == .playback ? 8 : 0)

                    DriveTelemetryDrawer(
                        detent: $panelDetent,
                        gnssQuality: gnssQuality,
                        routeLayer: $routeLayer,
                        showsAccuracyOverlay: showsAccuracyOverlay
                    )
                        .environmentObject(store)
                        .padding(.horizontal, 12)
                        .padding(.bottom, 10)
                }

                HStack {
                    Spacer()
                    DriveMapControlStack(
                        routeLayer: $routeLayer,
                        showsAccuracyOverlay: $showsAccuracyOverlay,
                        followsCurrentMarker: $followsCurrentMarker,
                        onViewportRefresh: {
                            viewportRefreshToken += 1
                        }
                    )
                    .padding(.trailing, 12)
                    .padding(.top, 74)
                }
                .frame(maxHeight: .infinity, alignment: .topTrailing)
            }
            .toolbar(.hidden, for: .navigationBar)
            .onChange(of: store.latestMotionEvent?.id) { _ in
                presentLatestMotionEventIfNeeded()
            }
            .task(id: headsUpMotionEvent?.id) {
                guard headsUpMotionEvent != nil else { return }
                let delayNanoseconds = UInt64(MotionEventHeadsUpPolicy.displayDurationSec * 1_000_000_000)
                do {
                    try await Task.sleep(nanoseconds: delayNanoseconds)
                } catch {
                    return
                }
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    withAnimation(.easeInOut(duration: 0.18)) {
                        headsUpMotionEvent = nil
                    }
                }
            }
        }
    }

    private func presentLatestMotionEventIfNeeded() {
        guard MotionEventHeadsUpPolicy.shouldPresent(
            latestID: store.latestMotionEvent?.id,
            displayedID: headsUpMotionEvent?.id
        ) else { return }
        withAnimation(.spring(response: 0.28, dampingFraction: 0.86)) {
            headsUpMotionEvent = store.latestMotionEvent
        }
    }
}

private struct TopMapStatusBar: View {
    @EnvironmentObject private var store: SensorStore
    let gnssQuality: GNSSQuality

    var body: some View {
        HStack(spacing: 8) {
            BrandMark(size: 30)
            Spacer(minLength: 4)
            CompactStatusDot(tint: streamTint)
            Text(store.streamMode.rawValue)
                .font(.caption.weight(.semibold))
                .lineLimit(1)
            Divider()
                .frame(height: 18)
            Image(systemName: gnssQuality.systemImage)
                .imageScale(.small)
                .foregroundStyle(gnssQuality.tint)
            Text(gnssQuality.compactTitle)
                .font(.caption.weight(.semibold))
                .lineLimit(1)
            Divider()
                .frame(height: 18)
            Image(systemName: store.ekfInitialized ? "checkmark.seal.fill" : "hourglass")
                .imageScale(.small)
                .foregroundStyle(store.ekfInitialized ? Color.accentColor : .orange)
            Text(store.ekfInitialized ? "Fusion" : "Align")
                .font(.caption.weight(.semibold))
                .lineLimit(1)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(.white.opacity(0.22), lineWidth: 1)
        )
    }

    private var streamTint: Color {
        if store.isRecording {
            return .red
        }
        if store.streamMode == .playback {
            return .purple
        }
        return .green
    }
}

private struct CompactStatusDot: View {
    let tint: Color

    var body: some View {
        Circle()
            .fill(tint)
            .frame(width: 8, height: 8)
            .overlay(
                Circle()
                    .stroke(tint.opacity(0.28), lineWidth: 4)
            )
            .accessibilityHidden(true)
    }
}

private struct DriveSessionInlineControls: View {
    @EnvironmentObject private var controls: SettingsControlModel

    private var canToggleLogging: Bool {
        controls.state.streamMode == .live && controls.state.isLiveSensorStreamRunning
    }

    var body: some View {
        HStack(spacing: 6) {
            Button {
                if controls.state.isLiveSensorStreamRunning {
                    controls.stopSensors()
                } else {
                    controls.startSensors()
                }
            } label: {
                DriveSessionInlineButtonLabel(
                    systemImage: controls.state.isLiveSensorStreamRunning ? "pause.fill" : "play.fill",
                    tint: controls.state.isLiveSensorStreamRunning ? .red : .accentColor
                )
            }
            .buttonStyle(.plain)
            .disabled(controls.state.streamMode == .playback)
            .opacity(controls.state.streamMode == .playback ? 0.45 : 1.0)
            .accessibilityLabel(controls.state.isLiveSensorStreamRunning ? "Stop data stream" : "Start data stream")

            Button {
                if controls.state.isRecording {
                    controls.stopRecording()
                } else {
                    controls.startRecording()
                }
            } label: {
                DriveSessionInlineButtonLabel(
                    systemImage: controls.state.isRecording ? "stop.fill" : "record.circle",
                    tint: controls.state.isRecording ? .red : .primary
                )
            }
            .buttonStyle(.plain)
            .disabled(!canToggleLogging && !controls.state.isRecording)
            .opacity((canToggleLogging || controls.state.isRecording) ? 1.0 : 0.45)
            .accessibilityLabel(controls.state.isRecording ? "Stop raw data logging" : "Start raw data logging")
        }
    }
}

private struct DriveSessionInlineButtonLabel: View {
    let systemImage: String
    let tint: Color

    var body: some View {
        Image(systemName: systemImage)
            .font(.system(size: 13, weight: .bold))
            .frame(width: 34, height: 34)
            .background(tint.opacity(0.12), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(tint.opacity(0.24), lineWidth: 1)
            )
        .foregroundStyle(tint)
        .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

private struct DriveMapControlStack: View {
    @Binding var routeLayer: RouteLayerSelection
    @Binding var showsAccuracyOverlay: Bool
    @Binding var followsCurrentMarker: Bool
    let onViewportRefresh: () -> Void

    var body: some View {
        VStack(spacing: 8) {
            Menu {
                Picker("Route Layer", selection: $routeLayer) {
                    ForEach(RouteLayerSelection.allCases) { layer in
                        Label(layer.displayTitle, systemImage: layer.systemImage)
                            .tag(layer)
                    }
                }
            } label: {
                MapControlButtonLabel(systemImage: routeLayer.systemImage, title: routeLayer.displayTitle)
            }
            .accessibilityLabel("Route layer")

            Button {
                showsAccuracyOverlay.toggle()
            } label: {
                MapControlButtonLabel(
                    systemImage: showsAccuracyOverlay ? "scope" : "scope",
                    title: showsAccuracyOverlay ? "Accuracy On" : "Accuracy"
                )
            }
            .buttonStyle(.plain)
            .foregroundStyle(showsAccuracyOverlay ? Color.accentColor : Color.primary)
            .accessibilityLabel(showsAccuracyOverlay ? "Hide accuracy overlay" : "Show accuracy overlay")

            Button {
                followsCurrentMarker.toggle()
                if followsCurrentMarker {
                    onViewportRefresh()
                }
            } label: {
                MapControlButtonLabel(
                    systemImage: followsCurrentMarker ? "location.fill.viewfinder" : "location.viewfinder",
                    title: followsCurrentMarker ? "Following" : "Follow"
                )
            }
            .buttonStyle(.plain)
            .foregroundStyle(followsCurrentMarker ? Color.accentColor : Color.primary)
            .accessibilityLabel(followsCurrentMarker ? "Stop following marker" : "Follow marker")
        }
    }
}

private struct MapControlButtonLabel: View {
    let systemImage: String
    let title: String

    var body: some View {
        Image(systemName: systemImage)
            .font(.system(size: 17, weight: .semibold))
            .frame(width: 42, height: 42)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(.white.opacity(0.22), lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.12), radius: 10, y: 4)
            .accessibilityLabel(title)
    }
}

private struct DriveTelemetryDrawer: View {
    @EnvironmentObject private var store: SensorStore
    @Binding var detent: DrivePanelDetent
    let gnssQuality: GNSSQuality
    @Binding var routeLayer: RouteLayerSelection
    let showsAccuracyOverlay: Bool
    @State private var selectedTab: DriveTelemetryTab = .metrics

    private var fusedSpeedMps: Double? {
        guard let latest = store.ekfVelocityHistory.last,
              let n = latest.x,
              let e = latest.y,
              let d = latest.z
        else {
            return nil
        }
        return sqrt(n * n + e * e + d * d)
    }

    private var health: FusionHealth {
        store.fusionHealth
    }

    private var statusProgress: Double {
        switch health.state {
        case .notReady, .initializing:
            return store.alignProgress.progress ?? 0.0
        case .running, .stable, .degraded, .degradedDeadReckoning, .awaitingGnssReseed:
            return health.fusedConfidence
        }
    }

    private var statusProgressLabel: String {
        switch health.state {
        case .notReady, .initializing:
            return "Alignment progress"
        case .running, .stable, .degraded, .degradedDeadReckoning, .awaitingGnssReseed:
            return "Fusion confidence"
        }
    }

    private var statusProgressTint: Color {
        switch health.state {
        case .notReady, .initializing:
            return .orange
        case .running, .stable, .degraded, .degradedDeadReckoning, .awaitingGnssReseed:
            if health.fusedConfidence >= 0.75 { return .accentColor }
            if health.fusedConfidence >= 0.45 { return .orange }
            return .red
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Capsule()
                    .fill(.secondary.opacity(0.32))
                    .frame(width: 38, height: 4)
                    .accessibilityHidden(true)
                Spacer()
                Button {
                    withAnimation(.spring(response: 0.28, dampingFraction: 0.86)) {
                        detent = detent.next()
                    }
                } label: {
                    Image(systemName: detent.systemImage)
                        .font(.caption.weight(.semibold))
                        .frame(width: 30, height: 26)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Resize telemetry drawer")
            }

            CollapsedDriveReadout(
                speedKmhText: DisplayUnitPolicy.speedKmhText(
                    fromMetersPerSecond: fusedSpeedMps ?? store.speedMps,
                    decimals: 1
                ),
                statusTitle: primaryDriveState,
                accuracyM: store.horizontalAccuracyM,
                progress: statusProgress,
                progressLabel: statusProgressLabel,
                tint: statusProgressTint
            )

            if detent != .collapsed {
                Picker("Telemetry", selection: $selectedTab) {
                    ForEach(DriveTelemetryTab.allCases) { tab in
                        Text(tab.rawValue).tag(tab)
                    }
                }
                .pickerStyle(.segmented)

                switch selectedTab {
                case .metrics:
                    DriveMetricGrid()
                        .environmentObject(store)
                case .streams:
                    VehicleMotionStreamPanel(
                        samples: store.vehicleMotionHistory,
                        isInitialized: store.ekfInitialized
                    )
                }
            }

            if detent == .expanded {
                RouteLegend(routeLayer: $routeLayer, showsAccuracyOverlay: showsAccuracyOverlay)
                RecentMotionEventsRow(events: Array(store.motionEvents.suffix(4)))
            }
        }
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(.white.opacity(0.25), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.14), radius: 16, y: 6)
    }

    private var primaryDriveState: String {
        switch health.state {
        case .stable:
            return "Stable"
        case .running:
            return "Ready"
        case .initializing:
            return "Aligning"
        case .degraded:
            return "Degraded"
        case .degradedDeadReckoning:
            return "Dead Reckoning"
        case .awaitingGnssReseed:
            return "Awaiting GNSS"
        case .notReady:
            return "Not Ready"
        }
    }
}

private struct CollapsedDriveReadout: View {
    @EnvironmentObject private var controls: SettingsControlModel

    let speedKmhText: String
    let statusTitle: String
    let accuracyM: Double?
    let progress: Double
    let progressLabel: String
    let tint: Color

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text(speedKmhText)
                    .font(.system(size: 28, weight: .semibold, design: .rounded))
                    .monospacedDigit()
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)
                Text("km/h")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
            }

            DriveSessionInlineControls()
                .environmentObject(controls)

            Spacer(minLength: 4)
            VStack(alignment: .trailing, spacing: 4) {
                Text(statusTitle)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)
                Text("GNSS \(format(accuracyM, decimals: 1)) m")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            StatusProgressRing(progress: progress, label: progressLabel, tint: tint)
        }
    }
}

private struct StatusProgressRing: View {
    let progress: Double
    let label: String
    let tint: Color

    private var normalizedProgress: Double {
        min(max(progress, 0.0), 1.0)
    }

    private var percent: Int {
        Int((normalizedProgress * 100.0).rounded())
    }

    var body: some View {
        ZStack {
            Circle()
                .stroke(.secondary.opacity(0.18), lineWidth: 4)
            Circle()
                .trim(from: 0.0, to: normalizedProgress)
                .stroke(tint, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                .rotationEffect(.degrees(-90))
            Text("\(percent)")
                .font(.caption2.weight(.semibold))
                .monospacedDigit()
        }
        .frame(width: 38, height: 38)
        .accessibilityLabel(label)
        .accessibilityValue("\(percent) percent")
    }
}

private struct DriveMetricGrid: View {
    @EnvironmentObject private var store: SensorStore

    private var hasTripStats: Bool {
        store.tripStatsSummary.sampleCount > 0
    }

    var body: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                MetricTile(title: "Pitch", value: format(store.ekfEulerHistory.last?.y, decimals: 1), unit: "deg")
                MetricTile(title: "Roll", value: format(store.ekfEulerHistory.last?.x, decimals: 1), unit: "deg")
                MetricTile(title: "Yaw", value: format(store.ekfEulerHistory.last?.z, decimals: 1), unit: "deg")
            }

            HStack(spacing: 8) {
                MetricTile(title: "Forward", value: DisplayUnitPolicy.velocityKmhText(fromMetersPerSecond: store.vehicleForwardMps, decimals: 1), unit: "km/h")
                MetricTile(title: "Lateral", value: DisplayUnitPolicy.velocityKmhText(fromMetersPerSecond: store.vehicleRightMps, decimals: 1), unit: "km/h")
                MetricTile(title: "Vertical", value: DisplayUnitPolicy.velocityKmhText(fromMetersPerSecond: store.vehicleDownMps, decimals: 1), unit: "km/h")
            }

            HStack(spacing: 8) {
                MetricTile(title: "Distance", value: tripDistanceText, unit: tripDistanceUnit)
                MetricTile(title: "Moving", value: tripMovingTimeText, unit: tripMovingTimeUnit)
                MetricTile(title: "Events", value: hasTripStats ? "\(store.tripStatsSummary.events.harshTotal)" : "-", unit: "harsh")
            }
        }
    }

    private var tripDistanceText: String {
        guard hasTripStats else { return "-" }
        return format(store.tripStatsSummary.distanceM / 1000.0, decimals: 2)
    }

    private var tripDistanceUnit: String {
        "km"
    }

    private var tripMovingTimeText: String {
        guard hasTripStats else { return "-" }
        return format(store.tripStatsSummary.movingDurationSec / 60.0, decimals: 1)
    }

    private var tripMovingTimeUnit: String {
        "min"
    }
}

private struct PlaybackMapControlPanel: View {
    @EnvironmentObject private var store: SensorStore

    var body: some View {
        if store.streamMode == .playback {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 10) {
                    Button {
                        store.stopPlayback()
                    } label: {
                        Image(systemName: "stop.fill")
                            .font(.caption.weight(.bold))
                            .frame(width: 32, height: 32)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.orange)
                    .accessibilityLabel("Stop playback")

                    VStack(alignment: .leading, spacing: 2) {
                        Text(store.activeSessionName ?? "Playback")
                            .font(.caption.weight(.semibold))
                            .lineLimit(1)
                        Text(playbackStatusText)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }

                    Spacer(minLength: 8)

                    Menu {
                        Picker("Playback Speed", selection: Binding(
                            get: { store.playbackSpeedMultiplier },
                            set: { store.setPlaybackSpeedMultiplier($0) }
                        )) {
                            ForEach(PlaybackSpeedPolicy.options, id: \.self) { speed in
                                Text(PlaybackSpeedPolicy.title(for: speed)).tag(speed)
                            }
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "speedometer")
                                .imageScale(.small)
                            Text(PlaybackSpeedPolicy.title(for: store.playbackSpeedMultiplier))
                                .monospacedDigit()
                            Image(systemName: "chevron.up.chevron.down")
                                .font(.caption2.weight(.semibold))
                        }
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 9)
                        .frame(height: 32)
                        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                    }
                    .accessibilityLabel("Playback speed")
                }

                ProgressView(value: store.replayProgress)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 9)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(.white.opacity(0.24), lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.12), radius: 12, y: 4)
            .accessibilityElement(children: .contain)
        }
    }

    private var playbackStatusText: String {
        let percent = Int((clampedReplayProgress * 100.0).rounded())
        guard let durationSec = activeSessionDurationSec else {
            return "\(percent)% complete"
        }
        let elapsedSec = clampedReplayProgress * durationSec
        return "\(formatPlaybackTime(elapsedSec)) / \(formatPlaybackTime(durationSec)) · \(percent)%"
    }

    private var clampedReplayProgress: Double {
        min(max(store.replayProgress, 0.0), 1.0)
    }

    private var activeSessionDurationSec: Double? {
        guard let activeSessionID = store.activeSessionID,
              let summary = store.recordedSessions.first(where: { $0.id == activeSessionID }),
              summary.durationSec.isFinite,
              summary.durationSec > 0.0
        else {
            return nil
        }
        return summary.durationSec
    }

    private func formatPlaybackTime(_ seconds: Double) -> String {
        let totalSeconds = max(0, Int(seconds.rounded(.down)))
        let hours = totalSeconds / 3_600
        let minutes = (totalSeconds % 3_600) / 60
        let secs = totalSeconds % 60
        if hours > 0 {
            return String(format: "%d:%02d:%02d", hours, minutes, secs)
        }
        return String(format: "%d:%02d", minutes, secs)
    }
}

private struct VehicleMotionStreamPanel: View {
    let samples: [SensorStore.TimedVec3Sample]
    let isInitialized: Bool

    private enum Trace: String, CaseIterable {
        case longitudinalAccel = "Longitudinal Accel"
        case yawRate = "Yaw Rate"

        var unit: String {
            switch self {
            case .longitudinalAccel: return "m/s²"
            case .yawRate: return "deg/s"
            }
        }

        var color: Color {
            switch self {
            case .longitudinalAccel: return .blue
            case .yawRate: return .purple
            }
        }

        var component: KeyPath<SensorStore.TimedVec3Sample, Double?> {
            switch self {
            case .longitudinalAccel: return \.x
            case .yawRate: return \.y
            }
        }

        var scale: Double {
            switch self {
            case .longitudinalAccel: return 3.0
            case .yawRate: return 45.0
            }
        }
    }

    private struct PlotSample: Identifiable {
        let id: String
        let trace: Trace
        let tSec: Double
        let normalizedValue: Double
    }

    private var windowSec: Double { 15.0 }
    private var tMax: Double { samples.last?.tSec ?? windowSec }
    private var xDomain: ClosedRange<Double> {
        (tMax - windowSec) ... tMax
    }

    private var plotSamples: [PlotSample] {
        let values = samples.filter { $0.tSec >= xDomain.lowerBound && $0.tSec <= xDomain.upperBound }
        let traces = Trace.allCases.flatMap { trace in
            values.enumerated().compactMap { index, sample -> PlotSample? in
                guard let value = sample[keyPath: trace.component], value.isFinite else { return nil }
                return PlotSample(
                    id: "\(trace.rawValue)-\(index)",
                    trace: trace,
                    tSec: sample.tSec,
                    normalizedValue: max(-1.0, min(1.0, value / trace.scale))
                )
            }
        }

        guard !traces.isEmpty else {
            return Trace.allCases.flatMap { trace in
                [
                    PlotSample(id: "\(trace.rawValue)-start", trace: trace, tSec: xDomain.lowerBound, normalizedValue: 0.0),
                    PlotSample(id: "\(trace.rawValue)-end", trace: trace, tSec: xDomain.upperBound, normalizedValue: 0.0)
                ]
            }
        }

        return traces
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 10) {
                ForEach(Trace.allCases, id: \.rawValue) { trace in
                    HStack(spacing: 5) {
                        Circle()
                            .fill(trace.color)
                            .frame(width: 7, height: 7)
                        Text(trace.rawValue)
                            .lineLimit(1)
                        Text("\(format(latestValue(for: trace), decimals: 1)) \(trace.unit)")
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    .font(.caption2.weight(.semibold))
                }
                Spacer(minLength: 0)
            }

            Chart {
                RuleMark(y: .value("Zero", 0.0))
                    .foregroundStyle(.secondary.opacity(0.28))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
                ForEach(plotSamples) { sample in
                    LineMark(
                        x: .value("Time", sample.tSec),
                        y: .value("Normalized", sample.normalizedValue),
                        series: .value("Trace", sample.trace.rawValue)
                    )
                    .foregroundStyle(sample.trace.color)
                    .interpolationMethod(.catmullRom)
                    .lineStyle(StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
                }
            }
            .chartXScale(domain: xDomain)
            .chartYScale(domain: -1.0 ... 1.0)
            .chartXAxis(.hidden)
            .chartYAxis(.hidden)
            .chartPlotStyle { plotArea in
                plotArea
                    .background(Color(.secondarySystemBackground).opacity(0.55))
                    .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            }
            .frame(height: 84)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(.white.opacity(0.20), lineWidth: 1)
        )
        .opacity(isInitialized ? 1.0 : 0.55)
        .accessibilityLabel(isInitialized ? "Vehicle motion stream plot" : "Vehicle motion stream plot waiting for EKF initialization")
    }

    private func latestValue(for trace: Trace) -> Double? {
        samples.reversed().compactMap { sample in
            let value = sample[keyPath: trace.component]
            return value?.isFinite == true ? value : nil
        }.first
    }
}

private struct MotionEventHeadsUp: View {
    let event: MotionEvent

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: event.kind.systemImage)
                .font(.caption.weight(.bold))
                .frame(width: 24, height: 24)
                .foregroundStyle(.white)
                .background(event.kind.color, in: Circle())
            VStack(alignment: .leading, spacing: 1) {
                Text(event.kind.displayTitle)
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
                Text(event.detailTitle)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer(minLength: 0)
            Text(formatEventTime(event.tSec))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(event.kind.color.opacity(0.35), lineWidth: 1)
        )
    }
}

private struct RecentMotionEventsRow: View {
    let events: [MotionEvent]

    var body: some View {
        if !events.isEmpty {
            VStack(alignment: .leading, spacing: 7) {
                Text("Events")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                HStack(spacing: 7) {
                    ForEach(events.reversed()) { event in
                        MotionEventChip(event: event)
                    }
                    Spacer(minLength: 0)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(.white.opacity(0.20), lineWidth: 1)
            )
        }
    }
}

private struct MotionEventChip: View {
    let event: MotionEvent

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: event.kind.systemImage)
                .imageScale(.small)
            Text(event.kind.displayTitle)
                .lineLimit(1)
        }
        .font(.caption2.weight(.semibold))
        .foregroundStyle(event.kind.color)
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(event.kind.color.opacity(0.12), in: Capsule())
    }
}

private struct DriveHeaderPanel: View {
    @EnvironmentObject private var store: SensorStore
    let gnssQuality: GNSSQuality

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                BrandMark()
                VStack(alignment: .leading, spacing: 2) {
                    Text("Motion Fusion")
                        .font(.headline.weight(.semibold))
                        .lineLimit(1)
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer(minLength: 8)
                StatusPill(title: gnssQuality.compactTitle, tint: gnssQuality.tint)
            }

            DriveStatusStrip(gnssQuality: gnssQuality)
                .environmentObject(store)
        }
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(.white.opacity(0.22), lineWidth: 1)
        )
    }

    private var subtitle: String {
        if !store.ekfInitialized {
            return "Automatic alignment"
        }
        if !store.ekfMountReady {
            return "Mount calibration pending"
        }
        return "Live vehicle motion"
    }
}

private struct BrandMark: View {
    var size: CGFloat = 38

    var body: some View {
        ZStack {
            Circle()
                .fill(Color.accentColor.opacity(0.18))
            Circle()
                .stroke(Color.accentColor.opacity(0.32), lineWidth: 1)
            Image(systemName: "location.north.line.fill")
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(Color.accentColor)
        }
        .frame(width: size, height: size)
        .accessibilityHidden(true)
    }
}

private struct DriveStatusStrip: View {
    @EnvironmentObject private var store: SensorStore
    let gnssQuality: GNSSQuality

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                StatusChip(
                    title: store.ekfInitialized ? "Initialized" : "Aligning",
                    systemImage: store.ekfInitialized ? "checkmark.seal.fill" : "hourglass",
                    tint: store.ekfInitialized ? .green : .orange
                )
                StatusChip(
                    title: store.ekfMountReady ? "Mount Ready" : "Mount Pending",
                    systemImage: store.ekfMountReady ? "mappin.and.ellipse" : "iphone.gen3.radiowaves.left.and.right",
                    tint: store.ekfMountReady ? .blue : .orange
                )
                StatusChip(
                    title: gnssQuality.title,
                    systemImage: gnssQuality.systemImage,
                    tint: gnssQuality.tint
                )
                if store.streamMode == .playback {
                    StatusChip(title: "Playback", systemImage: "play.circle.fill", tint: .purple)
                }
                if store.isRecording {
                    StatusChip(title: "Recording", systemImage: "record.circle", tint: .red)
                }
            }
            .padding(.vertical, 2)
        }
    }
}

private struct RouteLegend: View {
    @Binding var routeLayer: RouteLayerSelection
    var showsAccuracyOverlay: Bool = false

    var body: some View {
        HStack(spacing: 12) {
            Button {
                withAnimation(.spring(response: 0.24, dampingFraction: 0.88)) {
                    routeLayer = routeLayer.togglingFusedRoute()
                }
            } label: {
                LegendItem(
                    title: "Fused",
                    color: .accentColor,
                    width: 22,
                    isActive: routeLayer.showsFusedRoute
                )
            }
            .buttonStyle(.plain)
            .accessibilityLabel(routeLayer.showsFusedRoute ? "Hide fused route" : "Show fused route")

            Button {
                withAnimation(.spring(response: 0.24, dampingFraction: 0.88)) {
                    routeLayer = routeLayer.togglingGnssRoute()
                }
            } label: {
                LegendItem(
                    title: "GNSS",
                    color: .orange,
                    width: 16,
                    isActive: routeLayer.showsGnssRoute
                )
            }
            .buttonStyle(.plain)
            .accessibilityLabel(routeLayer.showsGnssRoute ? "Hide GNSS route" : "Show GNSS route")

            if showsAccuracyOverlay {
                LegendItem(title: "Accuracy", color: .green, width: 14, isActive: true)
            }
            Spacer(minLength: 0)
            Image(systemName: routeLayer.systemImage)
                .imageScale(.small)
                .foregroundStyle(.secondary)
        }
        .font(.caption.weight(.semibold))
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(.white.opacity(0.20), lineWidth: 1)
        )
    }
}

private struct LegendItem: View {
    let title: String
    let color: Color
    let width: CGFloat
    var isActive: Bool = true

    var body: some View {
        HStack(spacing: 6) {
            Capsule()
                .fill(color)
                .frame(width: width, height: 4)
            Text(title)
                .lineLimit(1)
                .foregroundStyle(.primary)
        }
        .opacity(isActive ? 1.0 : 0.42)
        .padding(.vertical, 4)
        .contentShape(Rectangle())
    }
}

private struct DriveMetricSheet: View {
    @EnvironmentObject private var store: SensorStore
    let gnssQuality: GNSSQuality

    private var fusedSpeedMps: Double? {
        guard let latest = store.ekfVelocityHistory.last,
              let n = latest.x,
              let e = latest.y,
              let d = latest.z
        else {
            return nil
        }
        return sqrt(n * n + e * e + d * d)
    }

    private var health: FusionHealth {
        store.fusionHealth
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text(primaryDriveState)
                        .font(.headline.weight(.semibold))
                    Text(secondaryDriveState)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                Spacer()
                StatusPill(title: store.ekfInitialized ? "Fusion" : "Align", tint: store.ekfInitialized ? .accentColor : .orange)
            }

            PrimaryMotionReadout(
                title: store.vehicleSegment?.displayTitle ?? "Vehicle",
                value: DisplayUnitPolicy.speedKmhText(fromMetersPerSecond: fusedSpeedMps ?? store.speedMps, decimals: 1),
                unit: "km/h",
                caption: "fused ground speed",
                confidence: health.fusedConfidence
            )

            HStack(spacing: 10) {
                MetricTile(title: "GNSS Speed", value: DisplayUnitPolicy.speedKmhText(fromMetersPerSecond: store.speedMps, decimals: 1), unit: "km/h")
                MetricTile(title: "Accuracy", value: format(store.horizontalAccuracyM, decimals: 1), unit: "m")
                MetricTile(title: "Age", value: formatAge(store.locationTimestamp), unit: "s")
            }

            HStack(spacing: 10) {
                MetricTile(title: "Forward", value: DisplayUnitPolicy.velocityKmhText(fromMetersPerSecond: store.vehicleForwardMps, decimals: 1), unit: "km/h")
                MetricTile(title: "Lateral", value: DisplayUnitPolicy.velocityKmhText(fromMetersPerSecond: store.vehicleRightMps, decimals: 1), unit: "km/h")
                MetricTile(title: "Vertical", value: DisplayUnitPolicy.velocityKmhText(fromMetersPerSecond: store.vehicleDownMps, decimals: 1), unit: "km/h")
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(.white.opacity(0.25), lineWidth: 1)
        )
    }

    private var primaryDriveState: String {
        switch health.state {
        case .stable:
            return "Stable"
        case .running:
            return "Running"
        case .initializing:
            return "Aligning phone mount"
        case .degraded:
            return "Degraded"
        case .degradedDeadReckoning:
            return "Dead reckoning"
        case .awaitingGnssReseed:
            return "Awaiting GNSS"
        case .notReady:
            return "Not ready"
        }
    }

    private var secondaryDriveState: String {
        switch health.state {
        case .stable:
            return "Fusion is converged enough to persist mount calibration."
        case .running:
            return "Fused route and vehicle-frame motion are active."
        case .initializing:
            return "Keep the phone fixed while alignment observes motion."
        case .degraded:
            return "Fusion is initialized, but current inputs or state checks are unhealthy."
        case .degradedDeadReckoning:
            return "Navigation is using degraded IMU dead reckoning until GNSS updates recover."
        case .awaitingGnssReseed:
            return "Calibration is retained, but navigation is waiting for GNSS reseed."
        case .notReady:
            return "Waiting for IMU and GNSS samples."
        }
    }
}

private struct ReviewView: View {
    @EnvironmentObject private var store: SensorStore

    var body: some View {
        NavigationStack {
            List {
                Section("Current Drive") {
                    valueRow("Route Points", "\(store.gnssRouteHistory.count)")
                    valueRow("Latest Speed (m/s)", format(store.speedMps, decimals: 2))
                    valueRow("Latest hAcc (m)", format(store.horizontalAccuracyM, decimals: 2))
                    valueRow("Initialized", yesNo(store.ekfInitialized))
                    valueRow("Mount Ready", yesNo(store.ekfMountReady))
                }

                Section("Completed Drives") {
                    if store.recordedSessions.isEmpty {
                        ContentUnavailableCompactView(
                            title: "No Saved Sessions",
                            systemImage: "tray",
                            message: "Recorded raw drives will appear here."
                        )
                    } else {
                        ForEach(store.recordedSessions) { summary in
                            RawSessionRow(summary: summary)
                                .environmentObject(store)
                        }
                    }
                }
            }
            .navigationTitle("Review")
            .toolbar {
                Button {
                    store.loadRecordedSessions()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .accessibilityLabel("Refresh sessions")
            }
            .onAppear {
                store.loadRecordedSessions()
            }
        }
    }
}

private struct RawSessionRow: View {
    @EnvironmentObject private var store: SensorStore
    let summary: RawSessionSummary

    var body: some View {
        let statusLabel = store.recordedSessionStatusLabel(for: summary)
        let detailMessage = store.recordedSessionDetailMessage(for: summary)
        let isInFlight = statusLabel == "Recording" || statusLabel == "Saving"
        let isActivePlayback = store.streamMode == .playback && store.activeSessionID == summary.id
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "waveform.path.ecg.rectangle")
                    .font(.title3)
                    .foregroundStyle(summary.isPendingSave ? .secondary : Color.accentColor)
                    .frame(width: 28)

                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Text(summary.name)
                            .font(.subheadline.weight(.semibold))
                            .lineLimit(2)
                        if let statusLabel {
                            Text(statusLabel)
                                .font(.caption2.weight(.semibold))
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(.quaternary, in: Capsule())
                        }
                    }
                    Text(summary.startTime.formatted(date: .abbreviated, time: .shortened))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    if let detailMessage {
                        Text(detailMessage)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(4)
                            .textSelection(.enabled)
                    }
                }

                Spacer(minLength: 8)

                if summary.isPendingSave {
                    if isInFlight {
                        ProgressView()
                            .controlSize(.small)
                            .frame(width: 44, height: 32)
                    } else {
                        Image(systemName: "exclamationmark.circle")
                            .font(.title3)
                            .foregroundStyle(.orange)
                            .frame(width: 44, height: 32)
                    }
                } else if isActivePlayback {
                    Button {
                        store.stopPlayback()
                    } label: {
                        Image(systemName: "stop.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.orange)
                    .accessibilityLabel("Stop playback")
                } else {
                    Button {
                        store.replaySession(summary)
                    } label: {
                        Image(systemName: "play.fill")
                    }
                    .buttonStyle(.bordered)
                    .disabled(summary.fileURL == nil || store.isRecording || store.streamMode == .playback)
                    .accessibilityLabel("Replay session")
                }
            }

            if isActivePlayback {
                HStack(spacing: 8) {
                    ProgressView(value: store.replayProgress)
                    Text(PlaybackSpeedPolicy.title(for: store.playbackSpeedMultiplier))
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
                .accessibilityLabel("Playback progress")
                .accessibilityValue("\(Int((store.replayProgress * 100.0).rounded())) percent")
            }

            HStack(spacing: 8) {
                SessionStat(title: "Duration", value: format(summary.durationSec, decimals: 1), unit: "s")
                SessionStat(title: "IMU", value: "\(summary.imuCount)", unit: "")
                SessionStat(title: "GNSS", value: "\(summary.gnssCount)", unit: "")
                SessionStat(title: "Baro", value: "\(summary.barometerCount)", unit: "")
            }
        }
        .swipeActions(edge: .trailing) {
            if !isInFlight {
                Button(role: .destructive) {
                    store.deleteSession(summary)
                } label: {
                    Label("Delete", systemImage: "trash")
                }
            }
        }
        .padding(.vertical, 4)
    }
}

private struct SessionStat: View {
    let title: String
    let value: String
    let unit: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            HStack(alignment: .firstTextBaseline, spacing: 2) {
                Text(value)
                    .font(.caption.weight(.semibold))
                    .monospacedDigit()
                    .lineLimit(1)
                if !unit.isEmpty {
                    Text(unit)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct DiagnosticsView: View {
    @EnvironmentObject private var store: SensorStore

#if DEBUG
    private var comparisonSnapshot: DeveloperComparisonSnapshot {
        DeveloperComparison.makeSnapshot(
            iosPositionM: iosPosition,
            fusedPositionM: fusedPosition,
            iosVelocityMps: iosVelocity,
            fusedVelocityMps: fusedVelocity,
            iosCourseDeg: comparableCourseDeg,
            fusedYawDeg: store.ekfEulerHistory.last?.z,
            iosAttitudeEulerDeg: iosAttitude,
            fusedEulerDeg: fusedEuler
        )
    }

    private var iosPosition: NavigationVectorNED? {
        guard let n = store.posNorthM, let e = store.posEastM, let d = store.posDownM else { return nil }
        return NavigationVectorNED(north: n, east: e, down: d)
    }

    private var fusedPosition: NavigationVectorNED? {
        guard let n = store.fusedPosNorthM, let e = store.fusedPosEastM, let d = store.fusedPosDownM else { return nil }
        return NavigationVectorNED(north: n, east: e, down: d)
    }

    private var iosVelocity: NavigationVectorNED? {
        guard let n = store.velNorthMps, let e = store.velEastMps else { return nil }
        return NavigationVectorNED(north: n, east: e, down: store.velDownMps ?? 0.0)
    }

    private var fusedVelocity: NavigationVectorNED? {
        guard let sample = store.ekfVelocityHistory.last,
              let n = sample.x,
              let e = sample.y,
              let d = sample.z
        else { return nil }
        return NavigationVectorNED(north: n, east: e, down: d)
    }

    private var comparableCourseDeg: Double? {
        guard let speed = store.speedMps, speed >= 2.0 else { return nil }
        return store.courseDeg
    }

    private var iosAttitude: NavigationVectorNED? {
        guard let sample = store.iosAttitudeEulerDeg,
              let roll = sample.x,
              let pitch = sample.y,
              let yaw = sample.z
        else { return nil }
        return NavigationVectorNED(north: roll, east: pitch, down: yaw)
    }

    private var fusedEuler: NavigationVectorNED? {
        guard let sample = store.ekfEulerHistory.last,
              let roll = sample.x,
              let pitch = sample.y,
              let yaw = sample.z
        else { return nil }
        return NavigationVectorNED(north: roll, east: pitch, down: yaw)
    }
#endif

    var body: some View {
        NavigationStack {
            List {
                Section("Fusion") {
                    valueRow("Initialized", yesNo(store.ekfInitialized))
                    valueRow("Mount Ready", yesNo(store.ekfMountReady))
                    valueRow("Confidence", format(store.fusionConfidence * 100.0, decimals: 0) + "%")
                    valueRow("Vehicle Segment", store.vehicleSegment?.displayTitle ?? "-")
                    valueRow("Fused Latitude", format(store.fusedLatitude, decimals: 7))
                    valueRow("Fused Longitude", format(store.fusedLongitude, decimals: 7))
                    valueRow("Fused Pn (m)", format(store.fusedPosNorthM, decimals: 2))
                    valueRow("Fused Pe (m)", format(store.fusedPosEastM, decimals: 2))
                    valueRow("Vehicle Forward (m/s)", format(store.vehicleForwardMps, decimals: 2))
                    valueRow("Vehicle Right (m/s)", format(store.vehicleRightMps, decimals: 2))
                    valueRow("Latest EKF Samples", "\(store.ekfVelocityHistory.count)")
                }

                Section("Fusion Profiling") {
                    valueRow("IMU Avg", profilingText(store.fusionProfiling.imu))
                    valueRow("GNSS Avg", profilingText(store.fusionProfiling.gnss))
                }

                Section("App Resources") {
                    valueRow("IMU Callback Rate", rateText(store.appResourceUsage.imuCallbackHz))
                    valueRow("GNSS Callback Rate", rateText(store.appResourceUsage.gnssCallbackHz))
                    valueRow("Fusion Enqueue Rate", rateText(store.appResourceUsage.fusionEnqueueHz))
                    valueRow("Fusion UI Rate", rateText(store.appResourceUsage.fusionUiPublishHz))
                    valueRow("Motion UI Rate", rateText(store.appResourceUsage.motionUiPublishHz))
                    valueRow("Road Event Rate", rateText(store.appResourceUsage.roadEventUpdateHz))
                    valueRow("Max Fusion Queue", "\(store.appResourceUsage.maxFusionQueueDepth)")
                    valueRow("Dropped Fusion Ops", "\(store.appResourceUsage.droppedFusionOperations)")
                    valueRow("Resident Memory", memoryText(store.appResourceUsage.residentMemoryMB))
                }

#if DEBUG
                DeveloperComparisonSection(snapshot: comparisonSnapshot)
#endif

                Section("Charts") {
                    ForEach(ChartKind.allCases) { kind in
                        NavigationLink(kind.title) {
                            ChartDetailView(kind: kind)
                                .environmentObject(store)
                        }
                    }
                }

                Section("Location") {
                    valueRow("Auth", authText(store.authorization))
                    valueRow("Latitude", format(store.latitude, decimals: 7))
                    valueRow("Longitude", format(store.longitude, decimals: 7))
                    valueRow("Altitude (m)", format(store.altitudeM, decimals: 2))
                    valueRow("Pn (m)", format(store.posNorthM, decimals: 2))
                    valueRow("Pe (m)", format(store.posEastM, decimals: 2))
                    valueRow("Pd (m)", format(store.posDownM, decimals: 2))
                    valueRow("Speed (m/s)", format(store.speedMps, decimals: 2))
                    valueRow("Course (deg)", format(store.courseDeg, decimals: 2))
                    valueRow("Vn (m/s)", format(store.velNorthMps, decimals: 2))
                    valueRow("Ve (m/s)", format(store.velEastMps, decimals: 2))
                    valueRow("Vd (m/s)", format(store.velDownMps, decimals: 2))
                    valueRow("hAcc (m)", format(store.horizontalAccuracyM, decimals: 2))
                    valueRow("vAcc (m)", format(store.verticalAccuracyM, decimals: 2))
                    valueRow("Timestamp", store.locationTimestamp?.formatted() ?? "-")
                }

                Section("IMU") {
                    valueRow("Accel X (m/s²)", format(store.motion.ax, decimals: 4))
                    valueRow("Accel Y (m/s²)", format(store.motion.ay, decimals: 4))
                    valueRow("Accel Z (m/s²)", format(store.motion.az, decimals: 4))
                    valueRow("Gyro X (rad/s)", format(store.motion.gx, decimals: 4))
                    valueRow("Gyro Y (rad/s)", format(store.motion.gy, decimals: 4))
                    valueRow("Gyro Z (rad/s)", format(store.motion.gz, decimals: 4))
                    valueRow("Motion Timestamp", store.motion.timestamp.formatted())
                }
            }
            .navigationTitle("Diagnostics")
        }
    }
}

#if DEBUG
private struct DeveloperComparisonSection: View {
    let snapshot: DeveloperComparisonSnapshot

    var body: some View {
        Section("Developer Comparison") {
            valueRow("Position dN / dE / dD", vectorText(snapshot.positionErrorM, decimals: 2, unit: "m"))
            valueRow("Horizontal Position Error", format(snapshot.horizontalPositionErrorM, decimals: 2) + " m")
            valueRow("Velocity dVn / dVe / dVd", vectorText(snapshot.velocityErrorMps, decimals: 2, unit: "m/s"))
            valueRow("Ground Speed Error", format(snapshot.groundSpeedErrorMps, decimals: 2) + " m/s")
            valueRow("Course Error", angleText(snapshot.courseErrorDeg))
            valueRow("Attitude dRoll / dPitch / dYaw", vectorText(snapshot.attitudeErrorDeg, decimals: 1, unit: "deg"))
            Text("Position/velocity references are iOS GNSS-derived states. Course compares only above 2 m/s. Attitude compares Core Motion device attitude against EKF attitude and is frame/mount sensitive.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func vectorText(_ vector: NavigationVectorNED?, decimals: Int, unit: String) -> String {
        guard let vector else { return "-" }
        return "\(format(vector.north, decimals: decimals)) / \(format(vector.east, decimals: decimals)) / \(format(vector.down, decimals: decimals)) \(unit)"
    }

    private func angleText(_ value: Double?) -> String {
        guard let value else { return "-" }
        return format(value, decimals: 1) + " deg"
    }
}
#endif

private struct SettingsView: View {
    @EnvironmentObject private var controls: SettingsControlModel
#if DEBUG
    @Binding var developerToolsEnabled: Bool
#endif

    var body: some View {
        let state = controls.state
        NavigationStack {
            List {
                Section("Permissions") {
                    valueRow("Location", authText(state.authorization))
                    valueRow("Motion", "Required")
                }

                Section("Stream") {
                    valueRow("Mode", state.streamStatusTitle)
                    Picker("Playback Speed", selection: Binding(
                        get: { controls.state.playbackSpeedMultiplier },
                        set: { controls.setPlaybackSpeedMultiplier($0) }
                    )) {
                        ForEach(PlaybackSpeedPolicy.options, id: \.self) { speed in
                            Text(PlaybackSpeedPolicy.title(for: speed)).tag(speed)
                        }
                    }
                    if let activeSessionName = state.activeSessionName {
                        valueRow("Session", activeSessionName)
                    }
                    if state.streamMode == .playback {
                        ProgressView(value: state.replayProgress)
                    }
                }

                Section("Event Audio") {
                    Picker("Harsh Behavior", selection: Binding(
                        get: { controls.state.harshBehaviorPreset },
                        set: { controls.setHarshBehaviorPreset($0) }
                    )) {
                        ForEach(HarshBehaviorPreset.allCases) { preset in
                            Text(preset.displayTitle).tag(preset)
                        }
                    }
                    .accessibilityIdentifier("harshBehaviorPresetPicker")
                    Picker("Audible Alerts", selection: Binding(
                        get: { controls.state.eventAudioSettings.mode },
                        set: { controls.setEventAudibleAlertMode($0) }
                    )) {
                        ForEach(EventAudibleAlertMode.allCases) { mode in
                            Text(mode.displayTitle).tag(mode)
                        }
                    }
                    Toggle(isOn: Binding(
                        get: { controls.state.eventAudioSettings.playDrivingAlertsInSilentMode },
                        set: { controls.setEventAlertsPlayInSilentMode($0) }
                    )) {
                        Text("Play Driving Alerts Anyway")
                    }
                    .disabled(state.eventAudioSettings.mode == .off)
                    Button {
                        controls.playTestEventAudioAlert()
                    } label: {
                        SettingsActionButtonLabel(
                            title: "Test Alert",
                            systemImage: "speaker.wave.2.circle",
                            tint: .accentColor
                        )
                    }
                    .buttonStyle(.plain)
                    .disabled(state.eventAudioSettings.mode == .off)
                    Text("Uses synthesized speech or chimes for driving events while the app is open. Playback mode can sound even when the phone is set to silent.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Raw Logging") {
                    valueRow("Saved Sessions", "\(state.savedSessionCount)")
                    valueRow("State", state.isRecording ? "Recording" : "Idle")
                    Button {
                        controls.refreshSessions()
                    } label: {
                        Label("Refresh Sessions", systemImage: "arrow.clockwise")
                    }
                }

                Section("Fusion") {
                    valueRow("Mode", controls.state.mountModeTitle)
                        .accessibilityIdentifier("mountModeRow")
                    Toggle(isOn: Binding(
                        get: { controls.state.mountMemorySettings.isEnabled },
                        set: { controls.setMountMemoryEnabled($0) }
                    )) {
                        Text("Remember Mount")
                    }
                    .accessibilityIdentifier("rememberMountToggle")
                    valueRow("Saved Mount", controls.state.savedMountTitle)
                        .accessibilityIdentifier("savedMountRow")
                    Button {
                        controls.clearRememberedMount()
                    } label: {
                        Label("Clear Remembered Mount", systemImage: "trash")
                    }
                    .disabled(controls.state.mountMemorySettings.savedCalibration == nil)
                    Text("When enabled, the app saves the aligned phone-to-vehicle mount and uses it on the next live or playback start. Auto alignment remains the fallback until a mount has been saved.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    valueRow("Map Layer", "GNSS + Fused")
                }

#if DEBUG
                Section("Developer Tools") {
                    Toggle("Diagnostics", isOn: $developerToolsEnabled)
                        .accessibilityIdentifier("developerDiagnosticsToggle")
                    Text("Shows internal comparison and diagnostic views in debug builds only.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
#endif
            }
            .navigationTitle("Settings")
        }
    }
}

private struct SettingsActionButtonLabel: View {
    let title: String
    let systemImage: String
    let tint: Color

    var body: some View {
        Label(title, systemImage: systemImage)
            .font(.body.weight(.semibold))
            .foregroundStyle(tint)
            .frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
            .contentShape(Rectangle())
    }
}

private struct StatusChip: View {
    let title: String
    let systemImage: String
    let tint: Color

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: systemImage)
                .imageScale(.small)
            Text(title)
                .font(.caption.weight(.semibold))
                .lineLimit(1)
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(.regularMaterial, in: Capsule())
        .overlay(
            Capsule()
                .stroke(tint.opacity(0.25), lineWidth: 1)
        )
    }
}

private struct StatusPill: View {
    let title: String
    let tint: Color

    var body: some View {
        Text(title)
            .font(.caption.weight(.semibold))
            .lineLimit(1)
            .foregroundStyle(tint)
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(tint.opacity(0.12), in: Capsule())
    }
}

private struct PrimaryMotionReadout: View {
    let title: String
    let value: String
    let unit: String
    let caption: String
    let confidence: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    HStack(alignment: .firstTextBaseline, spacing: 5) {
                        Text(value)
                            .font(.system(size: 36, weight: .semibold, design: .rounded))
                            .monospacedDigit()
                            .lineLimit(1)
                            .minimumScaleFactor(0.72)
                        Text(unit)
                            .font(.callout.weight(.semibold))
                            .foregroundStyle(.secondary)
                    }
                }
                Spacer(minLength: 10)
                Text(caption)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.trailing)
                    .lineLimit(2)
            }

            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(.secondary.opacity(0.18))
                    Capsule()
                        .fill(confidenceColor)
                        .frame(width: proxy.size.width * min(max(confidence, 0.0), 1.0))
                }
            }
            .frame(height: 5)
            .accessibilityLabel("Fusion confidence")
            .accessibilityValue("\(Int(confidence * 100.0)) percent")
        }
        .padding(12)
        .background(Color.accentColor.opacity(0.10), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    private var confidenceColor: Color {
        if confidence >= 0.75 {
            return .accentColor
        }
        if confidence >= 0.45 {
            return .orange
        }
        return .red
    }
}

private struct MetricTile: View {
    let title: String
    let value: String
    let unit: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(value)
                    .font(.system(.callout, design: .rounded).weight(.semibold))
                    .monospacedDigit()
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)
                if !unit.isEmpty {
                    Text(unit)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(.background.opacity(0.72), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

private struct ContentUnavailableCompactView: View {
    let title: String
    let systemImage: String
    let message: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Image(systemName: systemImage)
                .font(.title2)
                .foregroundStyle(.secondary)
            Text(title)
                .font(.headline)
            Text(message)
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 8)
    }
}

private enum GNSSQuality: Equatable {
    case good
    case fair
    case poor
    case missing

    init(horizontalAccuracyM: Double?, timestamp: Date?) {
        guard let horizontalAccuracyM, let timestamp else {
            self = .missing
            return
        }
        if Date().timeIntervalSince(timestamp) > 8.0 {
            self = .missing
        } else if horizontalAccuracyM <= 10.0 {
            self = .good
        } else if horizontalAccuracyM <= 30.0 {
            self = .fair
        } else {
            self = .poor
        }
    }

    var title: String {
        switch self {
        case .good: return "GNSS Good"
        case .fair: return "GNSS Fair"
        case .poor: return "Poor GNSS"
        case .missing: return "No GNSS Fix"
        }
    }

    var compactTitle: String {
        switch self {
        case .good: return "Good"
        case .fair: return "Fair"
        case .poor: return "Poor"
        case .missing: return "No Fix"
        }
    }

    var systemImage: String {
        switch self {
        case .good: return "location.fill"
        case .fair: return "location"
        case .poor: return "location.slash"
        case .missing: return "exclamationmark.triangle"
        }
    }

    var tint: Color {
        switch self {
        case .good: return .green
        case .fair: return .yellow
        case .poor: return .orange
        case .missing: return .red
        }
    }
}

private struct RawGNSSRoute {
    static func coordinates(
        currentLatitude: Double?,
        currentLongitude: Double?,
        currentNorthM: Double?,
        currentEastM: Double?,
        positionHistory: [SensorStore.TimedVec3Sample]
    ) -> [CLLocationCoordinate2D] {
        guard let currentLatitude,
              let currentLongitude,
              CLLocationCoordinate2DIsValid(CLLocationCoordinate2D(latitude: currentLatitude, longitude: currentLongitude))
        else {
            return []
        }

        guard let currentNorthM, let currentEastM else {
            return [CLLocationCoordinate2D(latitude: currentLatitude, longitude: currentLongitude)]
        }

        let scale = metersPerDegree(latitudeDeg: currentLatitude)
        guard abs(scale.latitude) > 0.001, abs(scale.longitude) > 0.001 else {
            return [CLLocationCoordinate2D(latitude: currentLatitude, longitude: currentLongitude)]
        }

        let referenceLatitude = currentLatitude - currentNorthM / scale.latitude
        let referenceLongitude = currentLongitude - currentEastM / scale.longitude
        let coordinates = positionHistory.compactMap { sample -> CLLocationCoordinate2D? in
            guard let northM = sample.x, let eastM = sample.y else { return nil }
            let coordinate = CLLocationCoordinate2D(
                latitude: referenceLatitude + northM / scale.latitude,
                longitude: referenceLongitude + eastM / scale.longitude
            )
            return CLLocationCoordinate2DIsValid(coordinate) ? coordinate : nil
        }

        let currentCoordinate = CLLocationCoordinate2D(latitude: currentLatitude, longitude: currentLongitude)
        guard !coordinates.isEmpty else {
            return [currentCoordinate]
        }

        if let last = coordinates.last,
           abs(last.latitude - currentLatitude) < 1.0e-8,
           abs(last.longitude - currentLongitude) < 1.0e-8 {
            return coordinates
        }
        return coordinates + [currentCoordinate]
    }

    private static func metersPerDegree(latitudeDeg: Double) -> (latitude: Double, longitude: Double) {
        let latRad = latitudeDeg * .pi / 180.0
        let metersPerDegLat = 111_132.92
            - 559.82 * cos(2.0 * latRad)
            + 1.175 * cos(4.0 * latRad)
            - 0.0023 * cos(6.0 * latRad)
        let metersPerDegLon = 111_412.84 * cos(latRad)
            - 93.5 * cos(3.0 * latRad)
            + 0.118 * cos(5.0 * latRad)
        return (metersPerDegLat, metersPerDegLon)
    }
}

private struct RawGNSSMapView: UIViewRepresentable {
    let gnssCoordinates: [CLLocationCoordinate2D]
    let fusedCoordinates: [CLLocationCoordinate2D]
    let currentCoordinate: CLLocationCoordinate2D?
    let fusedCurrentCoordinate: CLLocationCoordinate2D?
    let eventAnnotations: [MotionEvent]
    let currentHeadingDeg: Double?
    let horizontalAccuracyM: Double?
    let showAccuracyOverlay: Bool
    let followsCurrentMarker: Bool
    let followCameraVerticalOffset: CGFloat
    let viewportRefreshToken: Int

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> MKMapView {
        let mapView = MKMapView(frame: .zero)
        mapView.delegate = context.coordinator
        mapView.showsUserLocation = false
        mapView.showsCompass = true
        mapView.pointOfInterestFilter = .excludingAll
        mapView.isPitchEnabled = true
        mapView.isRotateEnabled = true
        mapView.mapType = .mutedStandard
        mapView.setRegion(
            MKCoordinateRegion(
                center: CLLocationCoordinate2D(latitude: 37.3349, longitude: -122.0090),
                span: MKCoordinateSpan(latitudeDelta: 0.03, longitudeDelta: 0.03)
            ),
            animated: false
        )
        return mapView
    }

    func updateUIView(_ mapView: MKMapView, context: Context) {
        AppPerformanceSignposts.interval("Map Update") {
            context.coordinator.currentHeadingDeg = currentHeadingDeg
            context.coordinator.isHeadingUp = followsCurrentMarker && currentHeadingDeg != nil
            context.coordinator.handleFollowModeChange(followsCurrentMarker: followsCurrentMarker)

            let routeKey = RouteKey(
                gnssCoordinates: gnssCoordinates,
                fusedCoordinates: fusedCoordinates,
                currentCoordinate: currentCoordinate,
                fusedCurrentCoordinate: fusedCurrentCoordinate
            )

            context.coordinator.updateAccuracyOverlay(
                on: mapView,
                coordinate: showAccuracyOverlay ? currentCoordinate : nil,
                horizontalAccuracyM: showAccuracyOverlay ? horizontalAccuracyM : nil
            )
            context.coordinator.updateRouteOverlays(
                on: mapView,
                gnssCoordinates: gnssCoordinates,
                fusedCoordinates: fusedCoordinates,
                routeKey: routeKey
            )
            context.coordinator.updateMarker(
                on: mapView,
                slot: .gnss,
                coordinate: currentCoordinate,
                title: "GNSS"
            )
            context.coordinator.updateMarker(
                on: mapView,
                slot: .fused,
                coordinate: fusedCurrentCoordinate,
                title: "Fused"
            )
            context.coordinator.updateEventAnnotations(on: mapView, events: eventAnnotations)

            let shouldForceRefit = context.coordinator.lastViewportRefreshToken != viewportRefreshToken
            if MapCameraPolicy.shouldRefit(
                isForced: shouldForceRefit,
                hasExistingViewport: context.coordinator.lastCameraViewportKey != nil,
                hasVisibleRoute: routeKey.hasVisibleRoute
            ) {
                context.coordinator.lastViewportRefreshToken = viewportRefreshToken
                context.coordinator.lastCameraViewportKey = routeKey.viewportKey
                setVisibleRoute(on: mapView, animated: shouldForceRefit)
            }
            if MapFollowPolicy.shouldApplyFollowCamera(
                followsCurrentMarker: followsCurrentMarker,
                isUserInteracting: context.coordinator.isUserInteracting(with: mapView),
                suspendedUntil: context.coordinator.followSuspendedUntil,
                now: Date()
            ),
               let followCoordinate = MapFollowPolicy.targetCoordinate(
                fusedCoordinate: fusedCurrentCoordinate,
                gnssCoordinate: currentCoordinate
               ) {
                context.coordinator.updateFollowCamera(
                    on: mapView,
                    coordinate: followCoordinate,
                    headingDeg: currentHeadingDeg,
                    verticalScreenOffset: followCameraVerticalOffset
                )
            }
        }
    }

    private func setVisibleRoute(on mapView: MKMapView, animated: Bool) {
        let visibleCoordinates = gnssCoordinates + fusedCoordinates
        if visibleCoordinates.count >= 2 {
            let polyline = MKPolyline(coordinates: visibleCoordinates, count: visibleCoordinates.count)
            var rect = polyline.boundingMapRect
            if let currentCoordinate {
                let currentPoint = MKMapPoint(currentCoordinate)
                rect = rect.union(MKMapRect(x: currentPoint.x, y: currentPoint.y, width: 1, height: 1))
            }
            if let fusedCurrentCoordinate {
                let currentPoint = MKMapPoint(fusedCurrentCoordinate)
                rect = rect.union(MKMapRect(x: currentPoint.x, y: currentPoint.y, width: 1, height: 1))
            }
            let minSide = max(rect.size.width, rect.size.height, 350.0)
            if rect.size.width < minSide {
                rect.origin.x -= (minSide - rect.size.width) * 0.5
                rect.size.width = minSide
            }
            if rect.size.height < minSide {
                rect.origin.y -= (minSide - rect.size.height) * 0.5
                rect.size.height = minSide
            }
            mapView.setVisibleMapRect(
                rect,
                edgePadding: UIEdgeInsets(top: 96, left: 32, bottom: 240, right: 32),
                animated: animated
            )
        } else if let currentCoordinate {
            mapView.setRegion(
                MKCoordinateRegion(
                    center: currentCoordinate,
                    span: MKCoordinateSpan(latitudeDelta: 0.008, longitudeDelta: 0.008)
                ),
                animated: animated
            )
        }
    }

    final class Coordinator: NSObject, MKMapViewDelegate {
        enum MarkerSlot {
            case gnss
            case fused
        }

        var currentHeadingDeg: Double?
        var isHeadingUp = false
        var lastRouteOverlayKey: RouteKey?
        var lastCameraViewportKey: String?
        var lastViewportRefreshToken: Int?
        var followSuspendedUntil: Date?
        private var lastFollowsCurrentMarker = false
        private var isApplyingProgrammaticCamera = false
        private var lastRouteOverlayUpdateDate: Date?
        private var lastGnssRoutePointCount: Int?
        private var lastFusedRoutePointCount: Int?
        private var accuracyOverlay: MKCircle?
        private var accuracyOverlayCoordinate: CLLocationCoordinate2D?
        private var accuracyOverlayRadiusM: Double?
        private let routeOverlayLayer = MapRouteOverlayLayer()
        private let gnssAnnotationLayer = MapAnnotationLayer()
        private let fusedAnnotationLayer = MapAnnotationLayer()
        private var eventAnnotationByID: [String: MotionEventMapAnnotation] = [:]

        func handleFollowModeChange(followsCurrentMarker: Bool) {
            if followsCurrentMarker && !lastFollowsCurrentMarker {
                followSuspendedUntil = nil
            }
            lastFollowsCurrentMarker = followsCurrentMarker
        }

        func isUserInteracting(with mapView: MKMapView) -> Bool {
            for subview in mapView.subviews {
                guard let recognizers = subview.gestureRecognizers else { continue }
                for recognizer in recognizers where recognizer.state == .began || recognizer.state == .changed {
                    return true
                }
            }
            return false
        }

        func updateFollowCamera(
            on mapView: MKMapView,
            coordinate: CLLocationCoordinate2D,
            headingDeg: Double?,
            verticalScreenOffset: CGFloat
        ) {
            guard CLLocationCoordinate2DIsValid(coordinate) else { return }
            let camera = mapView.camera
            camera.centerCoordinate = coordinate
            if let headingDeg {
                camera.heading = headingDeg
            }
            isApplyingProgrammaticCamera = true
            defer { isApplyingProgrammaticCamera = false }
            mapView.setCamera(camera, animated: false)
            guard verticalScreenOffset > 0.0, mapView.bounds.width > 1.0, mapView.bounds.height > 1.0 else {
                return
            }
            let desiredCenterPoint = CGPoint(
                x: mapView.bounds.midX,
                y: min(mapView.bounds.maxY, mapView.bounds.midY + verticalScreenOffset)
            )
            let adjustedCenter = mapView.convert(desiredCenterPoint, toCoordinateFrom: mapView)
            guard CLLocationCoordinate2DIsValid(adjustedCenter) else { return }
            camera.centerCoordinate = adjustedCenter
            mapView.setCamera(camera, animated: false)
        }

        func updateAccuracyOverlay(
            on mapView: MKMapView,
            coordinate: CLLocationCoordinate2D?,
            horizontalAccuracyM: Double?
        ) {
            guard let horizontalAccuracyM,
                  let coordinate,
                  horizontalAccuracyM > 0.0,
                  horizontalAccuracyM.isFinite
            else {
                if let accuracyOverlay {
                    mapView.removeOverlay(accuracyOverlay)
                    self.accuracyOverlay = nil
                }
                accuracyOverlayCoordinate = nil
                accuracyOverlayRadiusM = nil
                return
            }

            if let previousCoordinate = accuracyOverlayCoordinate,
               let previousRadius = accuracyOverlayRadiusM,
               abs(previousCoordinate.latitude - coordinate.latitude) < 1.0e-7,
               abs(previousCoordinate.longitude - coordinate.longitude) < 1.0e-7,
               abs(previousRadius - horizontalAccuracyM) < 0.25 {
                return
            }

            if let accuracyOverlay {
                mapView.removeOverlay(accuracyOverlay)
                self.accuracyOverlay = nil
            }

            let overlay = MKCircle(center: coordinate, radius: horizontalAccuracyM)
            accuracyOverlay = overlay
            accuracyOverlayCoordinate = coordinate
            accuracyOverlayRadiusM = horizontalAccuracyM
            mapView.addOverlay(overlay)
        }

        func updateRouteOverlays(
            on mapView: MKMapView,
            gnssCoordinates: [CLLocationCoordinate2D],
            fusedCoordinates: [CLLocationCoordinate2D],
            routeKey: RouteKey
        ) {
            guard routeKey != lastRouteOverlayKey else { return }
            let now = Date()
            let elapsed = lastRouteOverlayUpdateDate.map { now.timeIntervalSince($0) }
            guard MapRouteOverlayPolicy.shouldUpdate(
                previousGnssCount: lastGnssRoutePointCount,
                previousFusedCount: lastFusedRoutePointCount,
                nextGnssCount: gnssCoordinates.count,
                nextFusedCount: fusedCoordinates.count,
                elapsedSinceLastUpdateSec: elapsed
            ) else { return }

            lastRouteOverlayKey = routeKey
            lastRouteOverlayUpdateDate = now
            lastGnssRoutePointCount = gnssCoordinates.count
            lastFusedRoutePointCount = fusedCoordinates.count

            routeOverlayLayer.update(
                on: mapView,
                gnssCoordinates: gnssCoordinates,
                fusedCoordinates: fusedCoordinates
            )
        }

        func updateMarker(
            on mapView: MKMapView,
            slot: MarkerSlot,
            coordinate: CLLocationCoordinate2D?,
            title: String
        ) {
            let action = annotationLayer(for: slot).update(
                on: mapView,
                coordinate: coordinate,
                title: title
            )
            if action != .remove {
                updateMarkerView(on: mapView, slot: slot)
            }
        }

        private func annotationLayer(for slot: MarkerSlot) -> MapAnnotationLayer {
            switch slot {
            case .gnss: return gnssAnnotationLayer
            case .fused: return fusedAnnotationLayer
            }
        }

        private func updateMarkerView(on mapView: MKMapView, slot: MarkerSlot) {
            guard let annotation = annotationLayer(for: slot).annotation,
                  let view = mapView.view(for: annotation)
            else { return }
            guard let markerView = view as? PositionMarkerAnnotationView else { return }
            markerView.configure(slot: slot)
            if slot == .fused, !isHeadingUp, let currentHeadingDeg {
                view.transform = CGAffineTransform(rotationAngle: CGFloat(currentHeadingDeg * .pi / 180.0))
            } else {
                view.transform = .identity
            }
        }

        func updateEventAnnotations(on mapView: MKMapView, events: [MotionEvent]) {
            let displayable = events.compactMap { event -> (MotionEvent, CLLocationCoordinate2D)? in
                guard let coordinate = event.coordinate?.mapCoordinate else { return nil }
                return (event, coordinate)
            }
            let nextIDs = Set(displayable.map { $0.0.id })
            let staleIDs = Set(eventAnnotationByID.keys).subtracting(nextIDs)
            for id in staleIDs {
                if let annotation = eventAnnotationByID.removeValue(forKey: id) {
                    mapView.removeAnnotation(annotation)
                }
            }

            for (event, coordinate) in displayable {
                if let annotation = eventAnnotationByID[event.id] {
                    annotation.coordinate = coordinate
                } else {
                    let annotation = MotionEventMapAnnotation(event: event, coordinate: coordinate)
                    eventAnnotationByID[event.id] = annotation
                    mapView.addAnnotation(annotation)
                }
            }
        }

        func mapView(_ mapView: MKMapView, rendererFor overlay: MKOverlay) -> MKOverlayRenderer {
            if let circle = overlay as? MKCircle {
                let renderer = MKCircleRenderer(circle: circle)
                renderer.strokeColor = UIColor.systemOrange.withAlphaComponent(0.35)
                renderer.fillColor = UIColor.systemOrange.withAlphaComponent(0.08)
                renderer.lineWidth = 1
                return renderer
            }
            if let routeOverlay = overlay as? MapRouteOverlay {
                return MapRouteOverlayRenderer(routeOverlay: routeOverlay)
            }
            return MKOverlayRenderer(overlay: overlay)
        }

        func mapView(_ mapView: MKMapView, regionWillChangeAnimated animated: Bool) {
            guard !isApplyingProgrammaticCamera, isUserInteracting(with: mapView) else { return }
            followSuspendedUntil = MapFollowPolicy.followSuspendedUntil(afterUserInteractionAt: Date())
        }

        func mapView(_ mapView: MKMapView, viewFor annotation: MKAnnotation) -> MKAnnotationView? {
            guard !(annotation is MKUserLocation) else { return nil }
            if let eventAnnotation = annotation as? MotionEventMapAnnotation {
                let reuseID = "motion-event"
                let view = mapView.dequeueReusableAnnotationView(withIdentifier: reuseID) as? MotionEventAnnotationView
                    ?? MotionEventAnnotationView(annotation: annotation, reuseIdentifier: reuseID)
                view.annotation = annotation
                view.configure(kind: eventAnnotation.kind)
                view.displayPriority = .defaultHigh
                view.collisionMode = .circle
                view.zPriority = .defaultSelected
                return view
            }
            let slot: MarkerSlot = annotation.title == "Fused" ? .fused : .gnss
            let reuseID = slot == .fused ? "fused-position" : "gnss-position"
            let view = mapView.dequeueReusableAnnotationView(withIdentifier: reuseID) as? PositionMarkerAnnotationView
                ?? PositionMarkerAnnotationView(annotation: annotation, reuseIdentifier: reuseID)
            view.annotation = annotation
            view.configure(slot: slot)
            view.displayPriority = .required
            view.collisionMode = .circle
            view.zPriority = slot == .fused ? .max : .defaultUnselected
            if slot == .fused, !isHeadingUp, let currentHeadingDeg {
                view.transform = CGAffineTransform(rotationAngle: CGFloat(currentHeadingDeg * .pi / 180.0))
            } else {
                view.transform = .identity
            }
            return view
        }
    }

    final class MotionEventMapAnnotation: NSObject, MKAnnotation {
        let eventID: String
        let kind: MotionEvent.Kind
        let title: String?
        let subtitle: String?
        dynamic var coordinate: CLLocationCoordinate2D

        init(event: MotionEvent, coordinate: CLLocationCoordinate2D) {
            eventID = event.id
            kind = event.kind
            title = event.kind.displayTitle
            subtitle = event.detailTitle
            self.coordinate = coordinate
            super.init()
        }
    }

    final class MotionEventAnnotationView: MKAnnotationView {
        private let markerView = UIView()
        private let imageView = UIImageView()
        private var renderedKind: MotionEvent.Kind?

        override init(annotation: MKAnnotation?, reuseIdentifier: String?) {
            super.init(annotation: annotation, reuseIdentifier: reuseIdentifier)
            isOpaque = false
            canShowCallout = true
            bounds = CGRect(origin: .zero, size: CGSize(width: 30, height: 30))
            centerOffset = CGPoint(x: 0, y: -15)
            layer.shadowColor = UIColor.black.cgColor
            layer.shadowOpacity = 0.20
            layer.shadowRadius = 4
            layer.shadowOffset = CGSize(width: 0, height: 2)
            markerView.frame = bounds.insetBy(dx: 3, dy: 3)
            markerView.layer.cornerRadius = 12
            markerView.layer.borderColor = UIColor.white.withAlphaComponent(0.92).cgColor
            markerView.layer.borderWidth = 2
            addSubview(markerView)
            imageView.frame = markerView.frame.insetBy(dx: 5, dy: 5)
            imageView.contentMode = .scaleAspectFit
            imageView.tintColor = .white
            addSubview(imageView)
        }

        required init?(coder: NSCoder) {
            fatalError("init(coder:) has not been implemented")
        }

        func configure(kind: MotionEvent.Kind) {
            guard renderedKind != kind else { return }
            renderedKind = kind
            markerView.backgroundColor = kind.uiColor
            imageView.image = UIImage(systemName: kind.systemImage)
        }
    }

    final class PositionMarkerAnnotationView: MKAnnotationView {
        private var renderedSlot: Coordinator.MarkerSlot?

        func configure(slot: Coordinator.MarkerSlot) {
            guard renderedSlot != slot else { return }
            renderedSlot = slot
            image = Self.markerImage(for: slot)
            centerOffset = .zero
            canShowCallout = false
            bounds = CGRect(origin: .zero, size: image?.size ?? CGSize(width: 24, height: 24))
        }

        private static func markerImage(for slot: Coordinator.MarkerSlot) -> UIImage {
            let size = slot == .fused ? CGSize(width: 22, height: 22) : CGSize(width: 28, height: 28)
            let renderer = UIGraphicsImageRenderer(size: size)
            return renderer.image { context in
                let rect = CGRect(origin: .zero, size: size).insetBy(dx: 3, dy: 3)
                let cg = context.cgContext
                cg.setShadow(offset: CGSize(width: 0, height: 1), blur: 3, color: UIColor.black.withAlphaComponent(0.22).cgColor)

                switch slot {
                case .gnss:
                    UIColor.white.withAlphaComponent(0.92).setFill()
                    cg.fillEllipse(in: rect)
                    UIColor.systemOrange.setStroke()
                    cg.setLineWidth(3)
                    cg.strokeEllipse(in: rect)
                case .fused:
                    UIColor.systemBlue.setFill()
                    cg.fillEllipse(in: rect)
                    UIColor.white.setStroke()
                    cg.setLineWidth(2)
                    cg.strokeEllipse(in: rect)

                    let arrow = UIBezierPath()
                    let center = CGPoint(x: size.width * 0.5, y: size.height * 0.5)
                    arrow.move(to: CGPoint(x: center.x, y: center.y - 5.0))
                    arrow.addLine(to: CGPoint(x: center.x + 4.0, y: center.y + 4.5))
                    arrow.addLine(to: CGPoint(x: center.x, y: center.y + 2.2))
                    arrow.addLine(to: CGPoint(x: center.x - 4.0, y: center.y + 4.5))
                    arrow.close()
                    UIColor.white.setFill()
                    arrow.fill()
                }
            }
        }
    }

    struct RouteKey: Equatable {
        let gnssCount: Int
        let fusedCount: Int
        let gnssLastLatitude: Double?
        let gnssLastLongitude: Double?
        let fusedLastLatitude: Double?
        let fusedLastLongitude: Double?
        let currentLatitude: Double?
        let currentLongitude: Double?
        let fusedCurrentLatitude: Double?
        let fusedCurrentLongitude: Double?
        var hasVisibleRoute: Bool {
            gnssCount > 0 || fusedCount > 0 || currentLatitude != nil || fusedCurrentLatitude != nil
        }
        var viewportKey: String {
            [
                String(gnssCount),
                String(fusedCount),
                Self.roundedKey(gnssLastLatitude),
                Self.roundedKey(gnssLastLongitude),
                Self.roundedKey(fusedLastLatitude),
                Self.roundedKey(fusedLastLongitude),
                Self.roundedKey(currentLatitude),
                Self.roundedKey(currentLongitude),
                Self.roundedKey(fusedCurrentLatitude),
                Self.roundedKey(fusedCurrentLongitude)
            ].joined(separator: "|")
        }

        init(
            gnssCoordinates: [CLLocationCoordinate2D],
            fusedCoordinates: [CLLocationCoordinate2D],
            currentCoordinate: CLLocationCoordinate2D?,
            fusedCurrentCoordinate: CLLocationCoordinate2D?
        ) {
            gnssCount = gnssCoordinates.count
            fusedCount = fusedCoordinates.count
            gnssLastLatitude = gnssCoordinates.last?.latitude
            gnssLastLongitude = gnssCoordinates.last?.longitude
            fusedLastLatitude = fusedCoordinates.last?.latitude
            fusedLastLongitude = fusedCoordinates.last?.longitude
            currentLatitude = currentCoordinate?.latitude
            currentLongitude = currentCoordinate?.longitude
            fusedCurrentLatitude = fusedCurrentCoordinate?.latitude
            fusedCurrentLongitude = fusedCurrentCoordinate?.longitude
        }

        private static func roundedKey(_ value: Double?) -> String {
            guard let value else { return "-" }
            return String(format: "%.5f", value)
        }
    }
}

private struct ChartDetailView: View {
    @EnvironmentObject var store: SensorStore
    let kind: ChartKind
    @State private var xWindowSec: Double = 30.0
    @State private var xPanSec: Double = 0.0
    @State private var interactionAxis: ChartAxisMode = .xy

    var body: some View {
        VStack(spacing: 12) {
            Picker("Window", selection: $xWindowSec) {
                Text("10s").tag(10.0)
                Text("30s").tag(30.0)
                Text("60s").tag(60.0)
            }
            .pickerStyle(.segmented)

            Picker("Interaction", selection: $interactionAxis) {
                ForEach(ChartAxisMode.allCases) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            HStack {
                Button("Back To Live") { xPanSec = 0.0 }
                Spacer()
            }

            let labels = kind.axisLabels
            Vec3ChartPanel(
                xLabel: labels.0,
                yLabel: labels.1,
                zLabel: labels.2,
                samples: historyForKind(kind),
                interactionAxis: $interactionAxis,
                xWindowSec: $xWindowSec,
                xPanSec: $xPanSec
            )

            Spacer(minLength: 0)
        }
        .padding()
        .navigationTitle(kind.title)
        .navigationBarTitleDisplayMode(.inline)
    }

    private func historyForKind(_ kind: ChartKind) -> [SensorStore.TimedVec3Sample] {
        switch kind {
        case .nedPosition:
            return store.nedPositionHistory
        case .nedVelocity:
            return store.nedVelocityHistory
        case .imuAccel:
            return store.imuAccelHistory
        case .imuGyro:
            return store.imuGyroHistory
        case .ekfVelocity:
            return store.ekfVelocityHistory
        case .ekfEuler:
            return store.ekfEulerHistory
        case .ekfGyroBias:
            return store.ekfGyroBiasHistory
        case .ekfAccelBias:
            return store.ekfAccelBiasHistory
        }
    }
}

private struct Vec3ChartPanel: View {
    let xLabel: String
    let yLabel: String
    let zLabel: String
    let samples: [SensorStore.TimedVec3Sample]
    @Binding var interactionAxis: ChartAxisMode
    @Binding var xWindowSec: Double
    @Binding var xPanSec: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            TraceChartPanel(
                title: xLabel,
                color: .blue,
                samples: samples,
                component: \.x,
                interactionAxis: $interactionAxis,
                xWindowSec: $xWindowSec,
                xPanSec: $xPanSec
            )
            TraceChartPanel(
                title: yLabel,
                color: .green,
                samples: samples,
                component: \.y,
                interactionAxis: $interactionAxis,
                xWindowSec: $xWindowSec,
                xPanSec: $xPanSec
            )
            TraceChartPanel(
                title: zLabel,
                color: .orange,
                samples: samples,
                component: \.z,
                interactionAxis: $interactionAxis,
                xWindowSec: $xWindowSec,
                xPanSec: $xPanSec
            )
        }
    }
}

private struct TraceChartPanel: View {
    let title: String
    let color: Color
    let samples: [SensorStore.TimedVec3Sample]
    let component: KeyPath<SensorStore.TimedVec3Sample, Double?>
    @Binding var interactionAxis: ChartAxisMode
    @Binding var xWindowSec: Double
    @Binding var xPanSec: Double

    @State private var yZoom: Double = 1.0
    @State private var yPan: Double = 0.0
    @State private var pinchBaseXWindowSec: Double?
    @State private var pinchBaseYZoom: Double?
    @State private var dragBaseXPanSec: Double?
    @State private var dragBaseYPan: Double?

    private struct ScalarSample {
        let tSec: Double
        let value: Double
    }

    private var tMin: Double { samples.first?.tSec ?? 0.0 }
    private var tMax: Double { samples.last?.tSec ?? 0.0 }
    private var dataSpanSec: Double { max(tMax - tMin, 0.001) }
    private var clampedWindowSec: Double { min(max(xWindowSec, 5.0), 120.0) }
    private var maxPanSec: Double { max(0.0, dataSpanSec - clampedWindowSec) }
    private var clampedPanSec: Double { min(max(xPanSec, 0.0), maxPanSec) }
    private var xDomain: ClosedRange<Double> {
        let end = tMax - clampedPanSec
        let start = end - clampedWindowSec
        return start ... end
    }

    private var scalarSamples: [ScalarSample] {
        let windowed = samples.filter { $0.tSec >= xDomain.lowerBound && $0.tSec <= xDomain.upperBound }
        let maxPoints = 180
        let ekf: [SensorStore.TimedVec3Sample]
        if windowed.count > maxPoints {
            let strideN = max(1, windowed.count / maxPoints)
            ekf = stride(from: 0, to: windowed.count, by: strideN).map { windowed[$0] }
        } else {
            ekf = windowed
        }
        return ekf.compactMap { sample in
            guard let v = sample[keyPath: component] else { return nil }
            return ScalarSample(tSec: sample.tSec, value: v)
        }
    }

    private var yBaseDomain: ClosedRange<Double> {
        let values = scalarSamples.map(\.value)
        guard let minV = values.min(), let maxV = values.max() else {
            return -1.0 ... 1.0
        }
        let span = max(maxV - minV, 1e-6)
        let pad = span * 0.08
        return (minV - pad) ... (maxV + pad)
    }

    private var yDomain: ClosedRange<Double> {
        let baseCenter = (yBaseDomain.lowerBound + yBaseDomain.upperBound) * 0.5
        let baseSpan = max(yBaseDomain.upperBound - yBaseDomain.lowerBound, 1e-6)
        let span = baseSpan / max(yZoom, 1e-3)
        let center = baseCenter + yPan
        return (center - span * 0.5) ... (center + span * 0.5)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title)
                    .font(.caption)
                    .foregroundStyle(color)
                Spacer()
                Text(String(format: "Y %.1fx", yZoom))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Button("Reset Y") {
                    yZoom = 1.0
                    yPan = 0.0
                }
                .font(.caption2)
            }

            GeometryReader { geo in
                Chart {
                    ForEach(Array(scalarSamples.enumerated()), id: \.offset) { _, sample in
                        LineMark(
                            x: .value("Time (s)", sample.tSec),
                            y: .value(title, sample.value)
                        )
                        .foregroundStyle(color)
                    }
                }
                .chartXScale(domain: xDomain)
                .chartYScale(domain: yDomain)
                .chartLegend(.hidden)
                .simultaneousGesture(
                    MagnificationGesture()
                        .onChanged { scale in
                            if interactionAxis.allowsX {
                                let baseX = pinchBaseXWindowSec ?? xWindowSec
                                if pinchBaseXWindowSec == nil {
                                    pinchBaseXWindowSec = xWindowSec
                                }
                                xWindowSec = min(120.0, max(5.0, baseX / Double(scale)))
                                xPanSec = min(max(xPanSec, 0.0), maxPanSec)
                            }
                            if interactionAxis.allowsY {
                                let baseY = pinchBaseYZoom ?? yZoom
                                if pinchBaseYZoom == nil {
                                    pinchBaseYZoom = yZoom
                                }
                                yZoom = min(25.0, max(1.0, baseY * Double(scale)))
                            }
                        }
                        .onEnded { _ in
                            pinchBaseXWindowSec = nil
                            pinchBaseYZoom = nil
                        }
                )
                .simultaneousGesture(
                    DragGesture()
                        .onChanged { value in
                            if interactionAxis.allowsX {
                                let base = dragBaseXPanSec ?? xPanSec
                                if dragBaseXPanSec == nil {
                                    dragBaseXPanSec = xPanSec
                                }
                                let deltaSec = Double(value.translation.width / max(geo.size.width, 1.0)) * clampedWindowSec
                                xPanSec = min(max(base + deltaSec, 0.0), maxPanSec)
                            }
                            if interactionAxis.allowsY {
                                let base = dragBaseYPan ?? yPan
                                if dragBaseYPan == nil {
                                    dragBaseYPan = yPan
                                }
                                let currentSpan = max(yDomain.upperBound - yDomain.lowerBound, 1e-6)
                                let delta = Double(value.translation.height / max(geo.size.height, 1.0)) * currentSpan
                                yPan = base + delta
                            }
                        }
                        .onEnded { _ in
                            dragBaseXPanSec = nil
                            dragBaseYPan = nil
                        }
                )
                .onAppear {
                    xPanSec = min(max(xPanSec, 0.0), maxPanSec)
                }
                .onChange(of: samples.count) { _ in
                    xPanSec = min(max(xPanSec, 0.0), maxPanSec)
                }
                .onChange(of: xWindowSec) { _ in
                    xPanSec = min(max(xPanSec, 0.0), maxPanSec)
                }
            }
            .frame(height: 180)
        }
    }
}

private func valueRow(_ label: String, _ value: String) -> some View {
    HStack {
        Text(label)
        Spacer(minLength: 12)
        Text(value)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.trailing)
            .monospacedDigit()
    }
    .accessibilityElement(children: .combine)
    .accessibilityLabel("\(label), \(value)")
}

private func format(_ value: Double, decimals: Int) -> String {
    String(format: "%.*f", decimals, value)
}

private func format(_ value: Double?, decimals: Int) -> String {
    guard let value else { return "-" }
    return String(format: "%.*f", decimals, value)
}

private func profilingText(_ stats: FusionLoopProfilingStats) -> String {
    guard let averageMs = stats.averageMs else { return "-" }
    let lastMs = stats.lastMs ?? averageMs
    return String(format: "%.3f ms avg / %.3f last (%d)", averageMs, lastMs, stats.sampleCount)
}

private func rateText(_ value: Double?) -> String {
    guard let value else { return "-" }
    return String(format: "%.1f Hz", value)
}

private func memoryText(_ value: Double?) -> String {
    guard let value else { return "-" }
    return String(format: "%.1f MB", value)
}

private func formatAge(_ date: Date?) -> String {
    guard let date else { return "-" }
    return String(format: "%.1f", max(0.0, Date().timeIntervalSince(date)))
}

private func formatEventTime(_ tSec: Double) -> String {
    guard tSec.isFinite else { return "--:--" }
    let total = max(0, Int(tSec.rounded()))
    return String(format: "%d:%02d", total / 60, total % 60)
}

private func yesNo(_ value: Bool) -> String {
    value ? "yes" : "no"
}

private extension GeographicCoordinate {
    var mapCoordinate: CLLocationCoordinate2D? {
        guard isValidLatitudeLongitude else { return nil }
        let coordinate = CLLocationCoordinate2D(latitude: latitudeDeg, longitude: longitudeDeg)
        return CLLocationCoordinate2DIsValid(coordinate) ? coordinate : nil
    }
}

private extension MotionEvent {
    var detailTitle: String {
        switch kind {
        case .reverse:
            return "peak \(DisplayUnitPolicy.speedKmhText(fromMetersPerSecond: value, decimals: 1)) km/h"
        case .harshAcceleration:
            return "\(format(value, decimals: 1)) m/s²"
        case .harshBraking:
            return "\(format(value, decimals: 1)) m/s²"
        case .harshCornering:
            return "\(format(value, decimals: 1)) m/s² lateral"
        case .speedBump:
            return "\(format(value, decimals: 1))° peak pitch"
        case .roadShock:
            return "\(format(value, decimals: 1)) m/s² vertical"
        case .roughRoad:
            return "\(format(value, decimals: 2)) m/s² RMS"
        case .downhill, .uphill:
            return "\(format(value, decimals: 1))° pitch"
        case .gnssDegraded:
            return "location quality dropped"
        case .mountReady:
            return "automatic mount aligned"
        case .fusionReady:
            return "filter initialized"
        }
    }
}

private extension MotionEvent.Kind {
    var systemImage: String {
        MotionEventVisualPolicy.systemImage(for: self)
    }

    var color: Color {
        Color(uiColor)
    }

    var uiColor: UIColor {
        MotionEventVisualPolicy.uiColor(for: self)
    }
}

private func authText(_ status: CLAuthorizationStatus) -> String {
    switch status {
    case .notDetermined: return "notDetermined"
    case .restricted: return "restricted"
    case .denied: return "denied"
    case .authorizedAlways: return "authorizedAlways"
    case .authorizedWhenInUse: return "authorizedWhenInUse"
    @unknown default: return "unknown"
    }
}

#Preview {
    let store = SensorStore()
    ContentView()
        .environmentObject(store)
        .environmentObject(store.settingsControls)
}
