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
    @AppStorage("developerToolsEnabled") private var developerToolsEnabled = false
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
            if developerToolsEnabled {
                DiagnosticsView()
                    .tabItem {
                        Label("Diagnostics", systemImage: "waveform.path.ecg")
                    }
            }
#endif

            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gearshape")
                }
        }
    }
}

private struct DriveView: View {
    @EnvironmentObject private var store: SensorStore
    @State private var panelDetent: DrivePanelDetent = .collapsed
    @State private var routeLayer: RouteLayerSelection = .both
    @State private var showsAccuracyOverlay = false
    @State private var viewportRefreshToken = 0
    @State private var followsCurrentMarker = true

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
        return RawGNSSRoute.coordinates(
            currentLatitude: store.fusedLatitude,
            currentLongitude: store.fusedLongitude,
            currentNorthM: store.fusedPosNorthM,
            currentEastM: store.fusedPosEastM,
            positionHistory: store.fusedRouteHistory
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
                    currentHeadingDeg: currentMapHeadingDeg,
                    horizontalAccuracyM: store.horizontalAccuracyM,
                    showAccuracyOverlay: showsAccuracyOverlay,
                    followsCurrentMarker: followsCurrentMarker,
                    viewportRefreshToken: viewportRefreshToken
                )
                .ignoresSafeArea(edges: .top)

                VStack(spacing: 0) {
                    TopMapStatusBar(gnssQuality: gnssQuality)
                        .environmentObject(store)
                        .padding(.top, 8)
                        .padding(.horizontal, 12)

                    Spacer()

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
                    Spacer()
                        .frame(width: 0)
                }
                .frame(maxHeight: .infinity, alignment: .topTrailing)
            }
            .toolbar(.hidden, for: .navigationBar)
        }
    }
}

private struct TopMapStatusBar: View {
    @EnvironmentObject private var store: SensorStore
    let gnssQuality: GNSSQuality

    var body: some View {
        HStack(spacing: 8) {
            BrandMark(size: 30)
            Text("Motion Fusion")
                .font(.subheadline.weight(.semibold))
                .lineLimit(1)
            Spacer(minLength: 4)
            CompactStatusDot(tint: streamTint)
            Text(store.streamMode.rawValue)
                .font(.caption.weight(.semibold))
                .lineLimit(1)
            Divider()
                .frame(height: 18)
            Image(systemName: streamHealthImage)
                .imageScale(.small)
                .foregroundStyle(streamHealthTint)
            Text(store.streamHealth.shortTitle)
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

    private var streamHealthTint: Color {
        switch store.streamHealth.severity {
        case .nominal:
            return .green
        case .warning:
            return .orange
        case .critical:
            return .red
        }
    }

    private var streamHealthImage: String {
        switch store.streamHealth.severity {
        case .nominal:
            return "waveform"
        case .warning:
            return "exclamationmark.triangle.fill"
        case .critical:
            return "xmark.octagon.fill"
        }
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
        FusionHealth.evaluate(
            mountReady: store.ekfMountReady,
            initialized: store.ekfInitialized,
            gnssAccuracy: GnssAccuracy(
                horizontalAccuracyM: store.horizontalAccuracyM,
                verticalAccuracyM: store.verticalAccuracyM
            ),
            streamHealth: store.streamHealth
        )
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
                speedMps: fusedSpeedMps ?? store.speedMps,
                statusTitle: primaryDriveState,
                accuracyM: store.horizontalAccuracyM,
                confidence: health.fusedConfidence
            )

            if detent != .collapsed {
                DriveMetricGrid()
                    .environmentObject(store)
                PlaybackProgressStrip()
                    .environmentObject(store)
            }

            if detent == .expanded {
                RouteLegend(routeLayer: $routeLayer, showsAccuracyOverlay: showsAccuracyOverlay)
                ExpandedMotionRows()
                    .environmentObject(store)
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
        switch health.status {
        case .ready:
            return "Ready"
        case .aligning:
            return "Aligning"
        case .poorGNSS:
            return "Weak GNSS"
        case .needsMotion:
            return "Need Motion"
        case .alignmentIncomplete:
            return "Mount Pending"
        }
    }
}

private struct CollapsedDriveReadout: View {
    let speedMps: Double?
    let statusTitle: String
    let accuracyM: Double?
    let confidence: Double

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text(format(speedMps, decimals: 1))
                    .font(.system(size: 28, weight: .semibold, design: .rounded))
                    .monospacedDigit()
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)
                Text("m/s")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            VStack(alignment: .trailing, spacing: 4) {
                Text(statusTitle)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)
                Text("GNSS \(format(accuracyM, decimals: 1)) m")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            FusionConfidenceRing(confidence: confidence)
        }
    }
}

private struct FusionConfidenceRing: View {
    let confidence: Double

    var body: some View {
        ZStack {
            Circle()
                .stroke(.secondary.opacity(0.18), lineWidth: 4)
            Circle()
                .trim(from: 0.0, to: min(max(confidence, 0.0), 1.0))
                .stroke(color, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                .rotationEffect(.degrees(-90))
            Text("\(Int(confidence * 100.0))")
                .font(.caption2.weight(.semibold))
                .monospacedDigit()
        }
        .frame(width: 38, height: 38)
        .accessibilityLabel("Fusion confidence")
        .accessibilityValue("\(Int(confidence * 100.0)) percent")
    }

    private var color: Color {
        if confidence >= 0.75 { return .accentColor }
        if confidence >= 0.45 { return .orange }
        return .red
    }
}

private struct DriveMetricGrid: View {
    @EnvironmentObject private var store: SensorStore

    var body: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                MetricTile(title: "Pitch", value: format(store.ekfEulerHistory.last?.y, decimals: 1), unit: "deg")
                MetricTile(title: "Roll", value: format(store.ekfEulerHistory.last?.x, decimals: 1), unit: "deg")
                MetricTile(title: "Yaw", value: format(store.ekfEulerHistory.last?.z, decimals: 1), unit: "deg")
            }

            HStack(spacing: 8) {
                MetricTile(title: "Forward", value: format(store.vehicleForwardMps, decimals: 1), unit: "m/s")
                MetricTile(title: "Lateral", value: format(store.vehicleRightMps, decimals: 1), unit: "m/s")
                MetricTile(title: "Vertical", value: format(store.vehicleDownMps, decimals: 1), unit: "m/s")
            }
        }
    }
}

private struct PlaybackProgressStrip: View {
    @EnvironmentObject private var store: SensorStore

    var body: some View {
        if store.streamMode == .playback {
            HStack(spacing: 8) {
                Image(systemName: "play.circle.fill")
                    .foregroundStyle(.purple)
                ProgressView(value: store.replayProgress)
                Text("\(Int(store.replayProgress * 100.0))%")
                    .font(.caption)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .frame(width: 38, alignment: .trailing)
            }
        }
    }
}

private struct ExpandedMotionRows: View {
    @EnvironmentObject private var store: SensorStore

    var body: some View {
        VStack(spacing: 8) {
            valueRow("Fused Pn / Pe", "\(format(store.fusedPosNorthM, decimals: 1)) / \(format(store.fusedPosEastM, decimals: 1)) m")
            valueRow("Velocity Vn / Ve / Vd", "\(format(store.velNorthMps, decimals: 1)) / \(format(store.velEastMps, decimals: 1)) / \(format(store.velDownMps, decimals: 1)) m/s")
            valueRow("Route Samples", "\(store.gnssRouteHistory.count) GNSS, \(store.fusedRouteHistory.count) fused")
        }
        .font(.caption)
        .padding(.top, 2)
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
        FusionHealth.evaluate(
            mountReady: store.ekfMountReady,
            initialized: store.ekfInitialized,
            gnssAccuracy: GnssAccuracy(
                horizontalAccuracyM: store.horizontalAccuracyM,
                verticalAccuracyM: store.verticalAccuracyM
            ),
            streamHealth: store.streamHealth
        )
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
                value: format(fusedSpeedMps ?? store.speedMps, decimals: 1),
                unit: "m/s",
                caption: "fused ground speed",
                confidence: health.fusedConfidence
            )

            HStack(spacing: 10) {
                MetricTile(title: "GNSS Speed", value: format(store.speedMps, decimals: 1), unit: "m/s")
                MetricTile(title: "Accuracy", value: format(store.horizontalAccuracyM, decimals: 1), unit: "m")
                MetricTile(title: "Age", value: formatAge(store.locationTimestamp), unit: "s")
            }

            HStack(spacing: 10) {
                MetricTile(title: "Forward", value: format(store.vehicleForwardMps, decimals: 1), unit: "m/s")
                MetricTile(title: "Lateral", value: format(store.vehicleRightMps, decimals: 1), unit: "m/s")
                MetricTile(title: "Vertical", value: format(store.vehicleDownMps, decimals: 1), unit: "m/s")
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
        switch health.status {
        case .ready:
            return "Ready"
        case .aligning:
            return "Aligning phone mount"
        case .poorGNSS:
            return "Weak GNSS"
        case .needsMotion:
            return "Need vehicle motion"
        case .alignmentIncomplete:
            return "Mount calibration pending"
        }
    }

    private var secondaryDriveState: String {
        switch health.status {
        case .ready:
            return "Fused route and vehicle-frame motion are active."
        case .aligning:
            return "Keep the phone fixed while alignment observes motion."
        case .poorGNSS:
            return "Move to clearer sky or wait for location accuracy to recover."
        case .needsMotion:
            return "Drive straight briefly so the filter can settle."
        case .alignmentIncomplete:
            return "Motion is live while mount confidence improves."
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
                } else {
                    Button {
                        store.replaySession(summary)
                    } label: {
                        Image(systemName: "play.fill")
                    }
                    .buttonStyle(.bordered)
                    .disabled(summary.fileURL == nil || store.isRecording)
                    .accessibilityLabel("Replay session")
                }
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
    @AppStorage("developerToolsEnabled") private var developerToolsEnabled = false
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
                    if state.streamMode == .playback {
                        Button {
                            controls.stopPlayback()
                        } label: {
                            SettingsActionButtonLabel(
                                title: "Stop Playback",
                                systemImage: "pause.circle",
                                tint: .orange
                            )
                        }
                        .buttonStyle(.plain)
                    }
                    if state.streamMode == .live {
                        if state.isLiveSensorStreamRunning {
                            Button(role: .destructive) {
                                controls.stopSensors()
                            } label: {
                                SettingsActionButtonLabel(
                                    title: "Stop Sensors",
                                    systemImage: "stop.circle",
                                    tint: .red
                                )
                            }
                            .buttonStyle(.plain)
                        } else {
                            Button {
                                controls.startSensors()
                            } label: {
                                SettingsActionButtonLabel(
                                    title: "Start Sensors",
                                    systemImage: "play.circle",
                                    tint: .accentColor
                                )
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }

                Section("Raw Logging") {
                    valueRow("Saved Sessions", "\(state.savedSessionCount)")
                    if state.isRecording {
                        Button {
                            controls.stopRecording()
                        } label: {
                            Label("Stop Recording", systemImage: "stop.fill")
                        }
                        .tint(.red)
                    } else {
                        Button {
                            controls.startRecording()
                        } label: {
                            Label("Start Recording", systemImage: "record.circle")
                        }
                        .disabled(state.streamMode != .live || !state.isLiveSensorStreamRunning)
                    }
                    Button {
                        controls.refreshSessions()
                    } label: {
                        Label("Refresh Sessions", systemImage: "arrow.clockwise")
                    }
                }

                Section("Fusion") {
                    valueRow("Mode", "Ekf Auto Mount")
                    valueRow("Map Layer", "GNSS + Fused")
                }

#if DEBUG
                Section("Developer Tools") {
                    Toggle("Diagnostics", isOn: $developerToolsEnabled)
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
    let currentHeadingDeg: Double?
    let horizontalAccuracyM: Double?
    let showAccuracyOverlay: Bool
    let followsCurrentMarker: Bool
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
        context.coordinator.currentHeadingDeg = currentHeadingDeg
        context.coordinator.isHeadingUp = followsCurrentMarker && currentHeadingDeg != nil

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
        if followsCurrentMarker,
           let followCoordinate = MapFollowPolicy.targetCoordinate(
            fusedCoordinate: fusedCurrentCoordinate,
            gnssCoordinate: currentCoordinate
           ) {
            context.coordinator.updateFollowCamera(
                on: mapView,
                coordinate: followCoordinate,
                headingDeg: currentHeadingDeg
            )
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
        private var lastRouteOverlayUpdateDate: Date?
        private var lastGnssRoutePointCount: Int?
        private var lastFusedRoutePointCount: Int?
        private var accuracyOverlay: MKCircle?
        private var accuracyOverlayCoordinate: CLLocationCoordinate2D?
        private var accuracyOverlayRadiusM: Double?
        private let routeOverlayLayer = MapRouteOverlayLayer()
        private let gnssAnnotationLayer = MapAnnotationLayer()
        private let fusedAnnotationLayer = MapAnnotationLayer()

        func updateFollowCamera(
            on mapView: MKMapView,
            coordinate: CLLocationCoordinate2D,
            headingDeg: Double?
        ) {
            guard CLLocationCoordinate2DIsValid(coordinate) else { return }
            let camera = mapView.camera
            camera.centerCoordinate = coordinate
            if let headingDeg {
                camera.heading = headingDeg
            }
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

        func mapView(_ mapView: MKMapView, rendererFor overlay: MKOverlay) -> MKOverlayRenderer {
            if let circle = overlay as? MKCircle {
                let renderer = MKCircleRenderer(circle: circle)
                renderer.strokeColor = UIColor.systemOrange.withAlphaComponent(0.35)
                renderer.fillColor = UIColor.systemOrange.withAlphaComponent(0.08)
                renderer.lineWidth = 1
                return renderer
            }
            if let polyline = overlay as? MKPolyline {
                let renderer = MKPolylineRenderer(polyline: polyline)
                if polyline.title == "fused" {
                    renderer.strokeColor = UIColor.systemBlue.withAlphaComponent(0.90)
                    renderer.lineWidth = 5
                } else {
                    renderer.strokeColor = UIColor.systemOrange.withAlphaComponent(0.82)
                    renderer.lineWidth = 3
                }
                renderer.lineCap = .round
                renderer.lineJoin = .round
                return renderer
            }
            return MKOverlayRenderer(overlay: overlay)
        }

        func mapView(_ mapView: MKMapView, viewFor annotation: MKAnnotation) -> MKAnnotationView? {
            guard !(annotation is MKUserLocation) else { return nil }
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
}

private func format(_ value: Double, decimals: Int) -> String {
    String(format: "%.*f", decimals, value)
}

private func format(_ value: Double?, decimals: Int) -> String {
    guard let value else { return "-" }
    return String(format: "%.*f", decimals, value)
}

private func formatAge(_ date: Date?) -> String {
    guard let date else { return "-" }
    return String(format: "%.1f", max(0.0, Date().timeIntervalSince(date)))
}

private func yesNo(_ value: Bool) -> String {
    value ? "yes" : "no"
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
