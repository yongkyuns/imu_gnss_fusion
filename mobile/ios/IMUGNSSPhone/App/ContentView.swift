import Charts
import CoreLocation
import SwiftUI

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

struct ContentView: View {
    @EnvironmentObject var store: SensorStore

    var body: some View {
        NavigationStack {
            List {
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

                Section("Charts") {
                    ForEach(ChartKind.allCases) { kind in
                        NavigationLink(kind.title) {
                            ChartDetailView(kind: kind)
                                .environmentObject(store)
                        }
                    }
                }
            }
            .navigationTitle("IMU + GNSS")
        }
    }

    @ViewBuilder
    private func valueRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value)
                .foregroundStyle(.secondary)
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
        let reduced: [SensorStore.TimedVec3Sample]
        if windowed.count > maxPoints {
            let strideN = max(1, windowed.count / maxPoints)
            reduced = stride(from: 0, to: windowed.count, by: strideN).map { windowed[$0] }
        } else {
            reduced = windowed
        }
        return reduced.compactMap { sample in
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

#Preview {
    ContentView()
        .environmentObject(SensorStore())
}
