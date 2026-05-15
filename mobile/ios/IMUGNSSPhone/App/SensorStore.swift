import CoreLocation
import CoreMotion
import Foundation

private let g0Mps2 = 9.80665

final class SensorStore: NSObject, ObservableObject {
    struct TimedVec3Sample {
        let tSec: Double
        let x: Double?
        let y: Double?
        let z: Double?
    }

    struct MotionSample {
        var ax: Double = .zero
        var ay: Double = .zero
        var az: Double = .zero
        var gx: Double = .zero
        var gy: Double = .zero
        var gz: Double = .zero
        var timestamp: Date = .now
    }

    enum StreamMode: String {
        case live = "Live"
        case playback = "Playback"
    }

    @Published var authorization: CLAuthorizationStatus = .notDetermined
    @Published var latitude: Double?
    @Published var longitude: Double?
    @Published var altitudeM: Double?
    @Published var posNorthM: Double?
    @Published var posEastM: Double?
    @Published var posDownM: Double?
    @Published var speedMps: Double?
    @Published var courseDeg: Double?
    @Published var velNorthMps: Double?
    @Published var velEastMps: Double?
    @Published var velDownMps: Double?
    @Published var horizontalAccuracyM: Double?
    @Published var verticalAccuracyM: Double?
    @Published var locationTimestamp: Date?
    @Published var motion: MotionSample = .init()
    @Published var gnssRouteHistory: [TimedVec3Sample] = []
    @Published var fusedRouteHistory: [TimedVec3Sample] = []
    @Published var nedPositionHistory: [TimedVec3Sample] = []
    @Published var nedVelocityHistory: [TimedVec3Sample] = []
    @Published var imuAccelHistory: [TimedVec3Sample] = []
    @Published var imuGyroHistory: [TimedVec3Sample] = []
    @Published var ekfVelocityHistory: [TimedVec3Sample] = []
    @Published var ekfEulerHistory: [TimedVec3Sample] = []
    @Published var ekfGyroBiasHistory: [TimedVec3Sample] = []
    @Published var ekfAccelBiasHistory: [TimedVec3Sample] = []
    @Published var fusedPositionHistory: [TimedVec3Sample] = []
    @Published var ekfInitialized: Bool = false
    @Published var ekfMountReady: Bool = false
    @Published var fusedLatitude: Double?
    @Published var fusedLongitude: Double?
    @Published var fusedAltitudeM: Double?
    @Published var fusedPosNorthM: Double?
    @Published var fusedPosEastM: Double?
    @Published var fusedPosDownM: Double?
    @Published var vehicleForwardMps: Double?
    @Published var vehicleRightMps: Double?
    @Published var vehicleDownMps: Double?
    @Published var fusionConfidence: Double = 0.0
    @Published var vehicleSegment: VehicleMotionDisplay.Segment?
    @Published var streamMode: StreamMode = .live
    @Published var isRecording: Bool = false
    @Published var recordedSessions: [RawSessionSummary] = []
    @Published var activeSessionName: String?
    @Published var replayProgress: Double = 0.0
    @Published var streamHealth: StreamHealth = .starting
#if DEBUG
    @Published var iosAttitudeEulerDeg: TimedVec3Sample?
#endif

    private let locationManager = CLLocationManager()
    private let motionManager = CMMotionManager()
    private let altimeter = CMAltimeter()
    private let fusionEngine = FusionEngine()
    private let rawSessionStore = RawSessionFileStore()
    private let rawLogLock = NSLock()
    private let streamGenerationLock = NSLock()
    private let motionQueue = OperationQueue()
    private let fusionQueue = OperationQueue()
    private let barometerQueue = OperationQueue()
    private let recordingQueue = OperationQueue()
    private var nedReference: CLLocation?
    private var lastBarometerSample: (hM: Double, tS: TimeInterval)?
    private var filteredVerticalUpMps: Double?
    private var streamStartTime: Date?
    private var lastMotionPublishTS: TimeInterval?
    private var lastBaroPublishTS: TimeInterval?
    private var lastFusionUiPublishTS: TimeInterval?
    private var lastFusionSampleDate: Date?
    private var lastFusionImuSampleDate: Date?
    private var lastStreamHealthPublishTS: TimeInterval?
    private var lastImuUiSampleDate: Date?
    private var lastMotionErrorMessage: String?
    private var lastLocationErrorMessage: String?
    private var lastRecordingErrorMessage: String?
    private var activeRawLog: RawSessionLog?
    private var activeRawEvents: [RawSensorEventEnvelope] = []
    private var pendingRecordedSessions: [RawSessionSummary] = []
    private var playbackTask: Task<Void, Never>?
    private var streamHealthTask: Task<Void, Never>?
    private var lastRecordingCheckpointEventCount = 0
    private var lastRecordingCheckpointUptimeSec: TimeInterval = 0.0
    private var streamGeneration = 0
    private let motionPublishMinDtSec = 1.0 / 10.0
    private let baroPublishMinDtSec = 1.0 / 10.0
    private let fusionUiPublishMinDtSec = 1.0 / 10.0
    private let streamHealthPublishMinDtSec = 0.5
    private let replayProgressPublishMinDtSec = 1.0 / 15.0
    private let chartHistoryMaxCount = 240
    private let routeHistoryMaxCount = 21_600
    private let routeSampleMinDtSec = 1.0
    private let recordingCheckpointEventInterval = 5_000
    private let recordingCheckpointMinIntervalSec: TimeInterval = 30.0
    private let maximumPendingLiveFusionOperations = 50

    override init() {
        super.init()
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
        locationManager.distanceFilter = kCLDistanceFilterNone
        locationManager.activityType = .otherNavigation
        locationManager.pausesLocationUpdatesAutomatically = false
        locationManager.allowsBackgroundLocationUpdates = true
        locationManager.showsBackgroundLocationIndicator = true
        authorization = locationManager.authorizationStatus

        motionQueue.name = "imu_gnss_phone.motion.queue"
        motionQueue.qualityOfService = .userInitiated
        motionQueue.maxConcurrentOperationCount = 1

        fusionQueue.name = "imu_gnss_phone.fusion.queue"
        fusionQueue.qualityOfService = .userInitiated
        fusionQueue.maxConcurrentOperationCount = 1

        barometerQueue.name = "imu_gnss_phone.barometer.queue"
        barometerQueue.qualityOfService = .userInitiated
        barometerQueue.maxConcurrentOperationCount = 1

        recordingQueue.name = "imu_gnss_phone.recording.queue"
        recordingQueue.qualityOfService = .utility
        recordingQueue.maxConcurrentOperationCount = 1

        loadRecordedSessions()
    }

    deinit {
        checkpointRecording()
        streamHealthTask?.cancel()
    }

    func start() {
        let generation = advanceStreamGeneration()
        playbackTask?.cancel()
        playbackTask = nil
        fusionQueue.cancelAllOperations()
        streamMode = .live
        activeSessionName = nil
        replayProgress = 0.0
        startStreamHealthMonitor(generation: generation)
        nedReference = nil
        lastBarometerSample = nil
        filteredVerticalUpMps = nil
        streamStartTime = nil
        lastMotionPublishTS = nil
        lastBaroPublishTS = nil
        lastFusionUiPublishTS = nil
        lastStreamHealthPublishTS = nil
        lastImuUiSampleDate = nil
        lastMotionErrorMessage = nil
        lastLocationErrorMessage = nil
        lastRecordingErrorMessage = nil
        streamHealth = .starting
        lastStreamHealthPublishTS = nil
        lastImuUiSampleDate = nil
        lastMotionErrorMessage = nil
        lastLocationErrorMessage = nil
        lastRecordingErrorMessage = nil
        streamHealth = .starting
        gnssRouteHistory.removeAll(keepingCapacity: true)
        fusedRouteHistory.removeAll(keepingCapacity: true)
        nedPositionHistory.removeAll(keepingCapacity: true)
        nedVelocityHistory.removeAll(keepingCapacity: true)
        imuAccelHistory.removeAll(keepingCapacity: true)
        imuGyroHistory.removeAll(keepingCapacity: true)
        ekfVelocityHistory.removeAll(keepingCapacity: true)
        ekfEulerHistory.removeAll(keepingCapacity: true)
        ekfGyroBiasHistory.removeAll(keepingCapacity: true)
        ekfAccelBiasHistory.removeAll(keepingCapacity: true)
        fusedPositionHistory.removeAll(keepingCapacity: true)
        ekfInitialized = false
        ekfMountReady = false
        fusedLatitude = nil
        fusedLongitude = nil
        fusedAltitudeM = nil
        fusedPosNorthM = nil
        fusedPosEastM = nil
        fusedPosDownM = nil
        vehicleForwardMps = nil
        vehicleRightMps = nil
        vehicleDownMps = nil
        fusionConfidence = 0.0
        vehicleSegment = nil
#if DEBUG
        iosAttitudeEulerDeg = nil
#endif
        fusionQueue.addOperation { [weak self] in
            guard let self, self.isCurrentGeneration(generation) else { return }
            self.resetFusionEngineState()
        }

        if authorization == .notDetermined {
            locationManager.requestWhenInUseAuthorization()
        } else if authorization == .authorizedWhenInUse {
            locationManager.requestAlwaysAuthorization()
        }
        if authorization == .authorizedWhenInUse || authorization == .authorizedAlways {
            locationManager.startUpdatingLocation()
            locationManager.startUpdatingHeading()
        }

        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 1.0 / 100.0
            motionManager.startDeviceMotionUpdates(to: motionQueue) { [weak self] data, error in
                guard let self, self.isCurrentGeneration(generation) else { return }
                if let error {
                    self.publishStreamError(channel: .motion, error: error)
                    return
                }
                guard let data else {
                    self.publishStreamError(channel: .motion, message: "Device motion sample was unavailable.")
                    return
                }
                let timestampSec = data.timestamp
                let sampleDate = self.dateFromMotionTimestamp(timestampSec)
                let accel = self.bodySpecificForceMps2(from: data)
                let ax = accel.x
                let ay = accel.y
                let az = accel.z
                let gx = data.rotationRate.x
                let gy = data.rotationRate.y
                let gz = data.rotationRate.z
#if DEBUG
                let attitude = data.attitude
                let attitudeRollRad: Double? = attitude.roll
                let attitudePitchRad: Double? = attitude.pitch
                let attitudeYawRad: Double? = attitude.yaw
#else
                let attitudeRollRad: Double? = nil
                let attitudePitchRad: Double? = nil
                let attitudeYawRad: Double? = nil
#endif

                self.recordImuSample(
                    sampleDate: sampleDate,
                    sourceUptimeSec: timestampSec,
                    ax: ax,
                    ay: ay,
                    az: az,
                    gx: gx,
                    gy: gy,
                    gz: gz,
                    attitudeRollRad: attitudeRollRad,
                    attitudePitchRad: attitudePitchRad,
                    attitudeYawRad: attitudeYawRad
                )

                self.enqueueFusion(generation: generation, dropsWhenBacklogged: true) { store in
                    store.runEkfPredict(
                        sampleDate: sampleDate,
                        ax: ax,
                        ay: ay,
                        az: az,
                        gx: gx,
                        gy: gy,
                        gz: gz
                    )
                }

                if let last = self.lastMotionPublishTS, (data.timestamp - last) < self.motionPublishMinDtSec {
                    return
                }
                self.lastMotionPublishTS = data.timestamp
                Task { @MainActor in
                    guard self.isCurrentGeneration(generation) else { return }
                    self.publishImuSample(ax: ax, ay: ay, az: az, gx: gx, gy: gy, gz: gz, sampleDate: sampleDate)
#if DEBUG
                    self.publishIosAttitude(
                        rollRad: attitude.roll,
                        pitchRad: attitude.pitch,
                        yawRad: attitude.yaw,
                        sampleDate: sampleDate
                    )
#endif
                }
            }
        } else {
            lastMotionErrorMessage = "Device motion is unavailable on this device."
            updateStreamHealth(now: Date(), force: true)
        }

        if CMAltimeter.isRelativeAltitudeAvailable() {
            altimeter.startRelativeAltitudeUpdates(to: barometerQueue) { [weak self] data, error in
                guard let self, self.isCurrentGeneration(generation) else { return }
                if let error {
                    self.publishStreamError(channel: .barometer, error: error)
                    return
                }
                guard let data else { return }
                let hM = data.relativeAltitude.doubleValue
                let tS = data.timestamp

                var vDown: Double?
                if let prev = self.lastBarometerSample {
                    let dt = tS - prev.tS
                    if dt > 0.01 {
                        let vUp = (hM - prev.hM) / dt
                        let alpha = 0.20
                        let filteredUp = (self.filteredVerticalUpMps ?? vUp) + alpha * (vUp - (self.filteredVerticalUpMps ?? vUp))
                        self.filteredVerticalUpMps = filteredUp
                        vDown = -filteredUp
                    }
                }
                self.lastBarometerSample = (hM, tS)
                let sampleDate = self.dateFromMotionTimestamp(tS)
                self.recordBarometerSample(
                    sampleDate: sampleDate,
                    sourceUptimeSec: tS,
                    relativeAltitudeM: hM,
                    pressureKPa: data.pressure.doubleValue,
                    verticalDownVelocityMps: vDown
                )

                guard let vDown else { return }
                if let last = self.lastBaroPublishTS, (tS - last) < self.baroPublishMinDtSec {
                    return
                }
                self.lastBaroPublishTS = tS
                Task { @MainActor in
                    guard self.isCurrentGeneration(generation) else { return }
                    self.publishBarometerVelocity(vDown: vDown, sampleDate: sampleDate)
                }
            }
        } else {
            Task { @MainActor in
                self.velDownMps = nil
            }
        }
    }

    func stop() {
        _ = advanceStreamGeneration()
        finishRecordingIfNeeded()
        playbackTask?.cancel()
        playbackTask = nil
        streamHealthTask?.cancel()
        streamHealthTask = nil
        fusionQueue.cancelAllOperations()
        locationManager.stopUpdatingLocation()
        locationManager.stopUpdatingHeading()
        motionManager.stopDeviceMotionUpdates()
        altimeter.stopRelativeAltitudeUpdates()
        nedReference = nil
        lastBarometerSample = nil
        filteredVerticalUpMps = nil
        streamStartTime = nil
        lastMotionPublishTS = nil
        lastBaroPublishTS = nil
        lastFusionUiPublishTS = nil
        lastStreamHealthPublishTS = nil
        lastImuUiSampleDate = nil
        streamMode = .live
        activeSessionName = nil
        replayProgress = 0.0
        updateStreamHealth(now: Date(), force: true)
    }

    func startRecording() {
        guard streamMode == .live else { return }
        let log = RawSessionLog(
            name: "Drive \(Date().formatted(date: .abbreviated, time: .shortened))",
            startTime: Date()
        )
        rawLogLock.lock()
        activeRawLog = log
        activeRawEvents.removeAll(keepingCapacity: true)
        lastRecordingCheckpointEventCount = 0
        lastRecordingCheckpointUptimeSec = ProcessInfo.processInfo.systemUptime
        rawLogLock.unlock()
        isRecording = true
        activeSessionName = log.name
        updateStreamHealth(now: Date(), force: true)
    }

    func stopRecording() {
        finishRecordingIfNeeded()
    }

    func prepareForForegroundTracking() {
        locationManager.allowsBackgroundLocationUpdates = true
        locationManager.showsBackgroundLocationIndicator = false
        if authorization == .authorizedWhenInUse {
            locationManager.requestAlwaysAuthorization()
        }
    }

    func prepareForBackgroundTracking() {
        checkpointRecording()
        locationManager.allowsBackgroundLocationUpdates = true
        locationManager.showsBackgroundLocationIndicator = true
        if authorization == .authorizedAlways {
            locationManager.startUpdatingLocation()
            locationManager.startUpdatingHeading()
        }
    }

    func applicationWillTerminate() {
        checkpointRecording()
        recordingQueue.waitUntilAllOperationsAreFinished()
    }

    func loadRecordedSessions() {
        recordingQueue.addOperation { [weak self] in
            guard let self else { return }
            do {
                let summaries = try self.rawSessionStore.summaries()
                Task { @MainActor in
                    self.recordedSessions = self.mergedRecordedSessions(with: summaries)
                }
            } catch {
                print("Failed to load raw sessions: \(error)")
                Task { @MainActor in
                    self.recordedSessions = self.pendingRecordedSessions
                }
            }
        }
    }

    func deleteSession(_ summary: RawSessionSummary) {
        recordingQueue.addOperation { [weak self] in
            guard let self else { return }
            do {
                try self.rawSessionStore.delete(summary)
                let summaries = try self.rawSessionStore.summaries()
                Task { @MainActor in
                    self.pendingRecordedSessions.removeAll { $0.id == summary.id }
                    self.recordedSessions = self.mergedRecordedSessions(with: summaries)
                }
            } catch {
                print("Failed to delete raw session: \(error)")
            }
        }
    }

    func replaySession(_ summary: RawSessionSummary, speedMultiplier: Double = 10.0) {
        guard let fileURL = summary.fileURL else { return }
        let generation = advanceStreamGeneration()
        finishRecordingIfNeeded()
        locationManager.stopUpdatingLocation()
        locationManager.stopUpdatingHeading()
        motionManager.stopDeviceMotionUpdates()
        altimeter.stopRelativeAltitudeUpdates()
        playbackTask?.cancel()
        streamHealthTask?.cancel()
        fusionQueue.cancelAllOperations()

        let speed = max(speedMultiplier, 0.1)
        streamMode = .playback
        activeSessionName = summary.name
        replayProgress = 0.0
        resetRuntimeState()

        playbackTask = Task { [weak self] in
            guard let self else { return }
            do {
                let log = try self.rawSessionStore.load(from: fileURL)
                let events = try RawSessionTimeline.events(for: log)
                let duration = max(log.durationSec, 0.001)
                var previousElapsed = 0.0
                var lastProgressPublish = ProcessInfo.processInfo.systemUptime

                for event in events {
                    if Task.isCancelled { return }
                    let delay = max(0.0, event.elapsedSec - previousElapsed) / speed
                    if delay > 0.0 {
                        try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000.0))
                    }
                    previousElapsed = event.elapsedSec
                    self.applyReplayEvent(event, sessionStart: log.startTime, generation: generation)
                    let now = ProcessInfo.processInfo.systemUptime
                    if now - lastProgressPublish >= self.replayProgressPublishMinDtSec {
                        lastProgressPublish = now
                        await MainActor.run {
                            guard self.isCurrentGeneration(generation) else { return }
                            self.replayProgress = min(max(event.elapsedSec / duration, 0.0), 1.0)
                        }
                    }
                }
                await MainActor.run {
                    guard self.isCurrentGeneration(generation) else { return }
                    self.replayProgress = 1.0
                }
            } catch {
                print("Replay failed: \(error)")
                await MainActor.run {
                    guard self.isCurrentGeneration(generation) else { return }
                    self.streamMode = .live
                    self.activeSessionName = nil
                    self.replayProgress = 0.0
                }
            }
        }
    }

    func stopPlayback() {
        _ = advanceStreamGeneration()
        playbackTask?.cancel()
        playbackTask = nil
        fusionQueue.cancelAllOperations()
        streamMode = .live
        activeSessionName = nil
        replayProgress = 0.0
        start()
    }

    private func resetRuntimeState() {
        nedReference = nil
        lastBarometerSample = nil
        filteredVerticalUpMps = nil
        streamStartTime = nil
        lastMotionPublishTS = nil
        lastBaroPublishTS = nil
        lastFusionUiPublishTS = nil
        gnssRouteHistory.removeAll(keepingCapacity: true)
        fusedRouteHistory.removeAll(keepingCapacity: true)
        nedPositionHistory.removeAll(keepingCapacity: true)
        nedVelocityHistory.removeAll(keepingCapacity: true)
        imuAccelHistory.removeAll(keepingCapacity: true)
        imuGyroHistory.removeAll(keepingCapacity: true)
        ekfVelocityHistory.removeAll(keepingCapacity: true)
        ekfEulerHistory.removeAll(keepingCapacity: true)
        ekfGyroBiasHistory.removeAll(keepingCapacity: true)
        ekfAccelBiasHistory.removeAll(keepingCapacity: true)
        fusedPositionHistory.removeAll(keepingCapacity: true)
        ekfInitialized = false
        ekfMountReady = false
        latitude = nil
        longitude = nil
        altitudeM = nil
        posNorthM = nil
        posEastM = nil
        posDownM = nil
        speedMps = nil
        courseDeg = nil
        velNorthMps = nil
        velEastMps = nil
        velDownMps = nil
        horizontalAccuracyM = nil
        verticalAccuracyM = nil
        locationTimestamp = nil
        fusedLatitude = nil
        fusedLongitude = nil
        fusedAltitudeM = nil
        fusedPosNorthM = nil
        fusedPosEastM = nil
        fusedPosDownM = nil
        vehicleForwardMps = nil
        vehicleRightMps = nil
        vehicleDownMps = nil
        fusionConfidence = 0.0
        vehicleSegment = nil
#if DEBUG
        iosAttitudeEulerDeg = nil
#endif
        fusionQueue.cancelAllOperations()
        let generation = currentStreamGeneration()
        fusionQueue.addOperation { [weak self] in
            guard let self, self.isCurrentGeneration(generation) else { return }
            self.resetFusionEngineState()
        }
    }

    private func recordImuSample(
        sampleDate: Date,
        sourceUptimeSec: Double,
        ax: Double,
        ay: Double,
        az: Double,
        gx: Double,
        gy: Double,
        gz: Double,
        attitudeRollRad: Double?,
        attitudePitchRad: Double?,
        attitudeYawRad: Double?
    ) {
        guard let elapsed = recordingElapsed(for: sampleDate) else { return }
        appendRawEvent(.imu(
            RawImuSample(
                sourceUptimeSec: sourceUptimeSec,
                accelXMps2: ax,
                accelYMps2: ay,
                accelZMps2: az,
                gyroXRadps: gx,
                gyroYRadps: gy,
                gyroZRadps: gz,
                attitudeReferenceFrame: "default",
                attitudeRollRad: attitudeRollRad,
                attitudePitchRad: attitudePitchRad,
                attitudeYawRad: attitudeYawRad
            ),
            elapsedSec: elapsed,
            wallTime: sampleDate
        ))
    }

    private func recordGnssSample(
        sampleDate: Date,
        latitudeDeg: Double,
        longitudeDeg: Double,
        altitudeM: Double,
        horizontalAccuracyM: Double?,
        verticalAccuracyM: Double?,
        speedMps: Double?,
        courseDeg: Double?,
        speedAccuracyMps: Double?,
        courseAccuracyDeg: Double?,
        posN: Double?,
        posE: Double?,
        posD: Double?,
        velN: Double?,
        velE: Double?,
        velD: Double?
    ) {
        guard let elapsed = recordingElapsed(for: sampleDate) else { return }
        appendRawEvent(.gnss(
            RawGnssSample(
                latitudeDeg: latitudeDeg,
                longitudeDeg: longitudeDeg,
                altitudeM: altitudeM,
                horizontalAccuracyM: horizontalAccuracyM,
                verticalAccuracyM: verticalAccuracyM,
                speedMps: speedMps,
                courseDeg: courseDeg,
                speedAccuracyMps: speedAccuracyMps,
                courseAccuracyDeg: courseAccuracyDeg,
                positionNorthM: posN,
                positionEastM: posE,
                positionDownM: posD,
                velocityNorthMps: velN,
                velocityEastMps: velE,
                velocityDownMps: velD
            ),
            elapsedSec: elapsed,
            wallTime: sampleDate
        ))
    }

    private func recordBarometerSample(
        sampleDate: Date,
        sourceUptimeSec: Double,
        relativeAltitudeM: Double,
        pressureKPa: Double?,
        verticalDownVelocityMps: Double?
    ) {
        guard let elapsed = recordingElapsed(for: sampleDate) else { return }
        appendRawEvent(.barometer(
            RawBarometerSample(
                sourceUptimeSec: sourceUptimeSec,
                relativeAltitudeM: relativeAltitudeM,
                pressureKPa: pressureKPa,
                derivedVerticalVelocityDownMps: verticalDownVelocityMps
            ),
            elapsedSec: elapsed,
            wallTime: sampleDate
        ))
    }

    private func recordingElapsed(for date: Date) -> Double? {
        rawLogLock.lock()
        defer { rawLogLock.unlock() }
        guard let startTime = activeRawLog?.startTime else { return nil }
        return max(0.0, date.timeIntervalSince(startTime))
    }

    private func appendRawEvent(_ event: RawSensorEventEnvelope) {
        var checkpoint: RawSessionLog?
        rawLogLock.lock()
        if var log = activeRawLog {
            activeRawEvents.append(event)
            let eventCount = activeRawEvents.count
            let now = ProcessInfo.processInfo.systemUptime
            let hasEnoughEvents = eventCount - lastRecordingCheckpointEventCount >= recordingCheckpointEventInterval
            let hasEnoughTime = now - lastRecordingCheckpointUptimeSec >= recordingCheckpointMinIntervalSec
            if hasEnoughEvents && hasEnoughTime {
                lastRecordingCheckpointEventCount = eventCount
                lastRecordingCheckpointUptimeSec = now
                log.events = activeRawEvents
                checkpoint = log
            }
        }
        rawLogLock.unlock()

        if let checkpoint {
            saveRecordingCheckpoint(checkpoint)
        }
    }

    private func applyReplayEvent(_ event: RawReplayEvent, sessionStart: Date, generation: Int) {
        let sampleDate = sessionStart.addingTimeInterval(event.elapsedSec)

        switch event {
        case .imu(_, let sample):
            enqueueFusion(generation: generation) { store in
                store.runEkfPredict(
                    sampleDate: sampleDate,
                    ax: sample.accelXMps2,
                    ay: sample.accelYMps2,
                    az: sample.accelZMps2,
                    gx: sample.gyroXRadps,
                    gy: sample.gyroYRadps,
                    gz: sample.gyroZRadps
                )
            }
            Task { @MainActor in
                guard self.isCurrentGeneration(generation) else { return }
                self.publishImuSample(
                    ax: sample.accelXMps2,
                    ay: sample.accelYMps2,
                    az: sample.accelZMps2,
                    gx: sample.gyroXRadps,
                    gy: sample.gyroYRadps,
                    gz: sample.gyroZRadps,
                    sampleDate: sampleDate
                )
#if DEBUG
                if let rollRad = sample.attitudeRollRad,
                   let pitchRad = sample.attitudePitchRad,
                   let yawRad = sample.attitudeYawRad {
                    self.publishIosAttitude(
                        rollRad: rollRad,
                        pitchRad: pitchRad,
                        yawRad: yawRad,
                        sampleDate: sampleDate
                    )
                }
#endif
            }

        case .gnss(_, let sample):
            let fallbackPosition = fallbackNedPosition(
                latitudeDeg: sample.latitudeDeg,
                longitudeDeg: sample.longitudeDeg,
                altitudeM: sample.altitudeM,
                horizontalAccuracyM: sample.horizontalAccuracyM,
                verticalAccuracyM: sample.verticalAccuracyM,
                sampleDate: sampleDate
            )
            let fallbackVelocity = fallbackNedVelocity(speedMps: sample.speedMps, courseDeg: sample.courseDeg)
            let posN = sample.positionNorthM ?? fallbackPosition.n
            let posE = sample.positionEastM ?? fallbackPosition.e
            let posD = sample.positionDownM ?? fallbackPosition.d
            let velN = sample.velocityNorthMps ?? fallbackVelocity.n
            let velE = sample.velocityEastMps ?? fallbackVelocity.e
            let velD = sample.velocityDownMps ?? velDownMps
            let hAcc = sample.horizontalAccuracyM ?? 25.0
            let vAcc = sample.verticalAccuracyM ?? 50.0

            enqueueFusion(generation: generation) { store in
                store.runEkfFuseGps(
                    sampleDate: sampleDate,
                    latitudeDeg: sample.latitudeDeg,
                    longitudeDeg: sample.longitudeDeg,
                    altitudeM: sample.altitudeM,
                    posN: posN,
                    posE: posE,
                    posD: posD,
                    velN: velN,
                    velE: velE,
                    velD: velD,
                    hAcc: hAcc,
                    vAcc: vAcc,
                    courseDeg: sample.courseDeg,
                    speedAccuracyMps: sample.speedAccuracyMps
                )
            }
            Task { @MainActor in
                guard self.isCurrentGeneration(generation) else { return }
                self.publishGnssSample(
                    sampleDate: sampleDate,
                    latitudeDeg: sample.latitudeDeg,
                    longitudeDeg: sample.longitudeDeg,
                    altitudeM: sample.altitudeM,
                    posN: posN,
                    posE: posE,
                    posD: posD,
                    speedMps: sample.speedMps,
                    courseDeg: sample.courseDeg,
                    velN: velN,
                    velE: velE,
                    velD: velD,
                    horizontalAccuracyM: sample.horizontalAccuracyM,
                    verticalAccuracyM: sample.verticalAccuracyM
                )
            }

        case .barometer(_, let sample):
            guard let vDown = sample.derivedVerticalVelocityDownMps else { return }
            Task { @MainActor in
                guard self.isCurrentGeneration(generation) else { return }
                self.publishBarometerVelocity(vDown: vDown, sampleDate: sampleDate)
            }
        }
    }

    private func fallbackNedPosition(
        latitudeDeg: Double,
        longitudeDeg: Double,
        altitudeM: Double,
        horizontalAccuracyM: Double?,
        verticalAccuracyM: Double?,
        sampleDate: Date
    ) -> (n: Double?, e: Double?, d: Double?) {
        let coordinate = CLLocationCoordinate2D(latitude: latitudeDeg, longitude: longitudeDeg)
        guard CLLocationCoordinate2DIsValid(coordinate) else { return (nil, nil, nil) }
        let location = CLLocation(
            coordinate: coordinate,
            altitude: altitudeM,
            horizontalAccuracy: horizontalAccuracyM ?? 25.0,
            verticalAccuracy: verticalAccuracyM ?? 50.0,
            timestamp: sampleDate
        )
        return computeNEDPositionFromGnss(current: location)
    }

    private func fallbackNedVelocity(speedMps: Double?, courseDeg: Double?) -> (n: Double?, e: Double?) {
        guard let velocity = GnssVelocityResolver.horizontalVelocity(
            speedMps: speedMps,
            courseDeg: courseDeg
        ) else { return (nil, nil) }
        return (velocity.northMps, velocity.eastMps)
    }

    private func finishRecordingIfNeeded() {
        rawLogLock.lock()
        var log = activeRawLog
        if log != nil {
            log?.events = activeRawEvents
        }
        activeRawLog = nil
        activeRawEvents.removeAll(keepingCapacity: true)
        lastRecordingCheckpointEventCount = 0
        rawLogLock.unlock()

        guard let log else {
            isRecording = false
            if streamMode == .live {
                activeSessionName = nil
            }
            return
        }
        guard !log.events.isEmpty else {
            isRecording = false
            if streamMode == .live {
                activeSessionName = nil
            }
            return
        }
        let pendingSummary = log.summary(fileURL: nil)
        pendingRecordedSessions.removeAll { $0.id == pendingSummary.id }
        pendingRecordedSessions.insert(pendingSummary, at: 0)
        recordedSessions = mergedRecordedSessions(with: recordedSessions.filter { !$0.isPendingSave })
        saveRecording(log, reason: "save raw session", refreshSummaries: true)
        isRecording = false
        if streamMode == .live {
            activeSessionName = nil
        }
    }

    func checkpointRecording() {
        rawLogLock.lock()
        var log = activeRawLog
        if log != nil {
            log?.events = activeRawEvents
        }
        lastRecordingCheckpointEventCount = activeRawEvents.count
        lastRecordingCheckpointUptimeSec = ProcessInfo.processInfo.systemUptime
        rawLogLock.unlock()

        guard let log, !log.events.isEmpty else { return }
        saveRecordingCheckpoint(log)
    }

    private func saveRecordingCheckpoint(_ log: RawSessionLog) {
        saveRecording(log, reason: "checkpoint raw session", refreshSummaries: false)
    }

    private func saveRecording(_ log: RawSessionLog, reason: String, refreshSummaries: Bool) {
        recordingQueue.addOperation { [weak self] in
            guard let self else { return }
            do {
                _ = try self.rawSessionStore.save(log)
                Task { @MainActor in
                    self.lastRecordingErrorMessage = nil
                    self.updateStreamHealth(now: Date(), force: true)
                }
                if refreshSummaries {
                    Task { @MainActor in
                        self.pendingRecordedSessions.removeAll { $0.id == log.id }
                        self.loadRecordedSessions()
                    }
                }
            } catch {
                print("Failed to \(reason): \(error)")
                Task { @MainActor in
                    self.lastRecordingErrorMessage = error.localizedDescription
                    if refreshSummaries {
                        self.pendingRecordedSessions.removeAll { $0.id == log.id }
                        self.recordedSessions = self.mergedRecordedSessions(
                            with: self.recordedSessions.filter { !$0.isPendingSave }
                        )
                    }
                    self.updateStreamHealth(now: Date(), force: true)
                }
            }
        }
    }

    private func mergedRecordedSessions(with savedSummaries: [RawSessionSummary]) -> [RawSessionSummary] {
        let savedIDs = Set(savedSummaries.map(\.id))
        let pending = pendingRecordedSessions.filter { !savedIDs.contains($0.id) }
        return (pending + savedSummaries).sorted { lhs, rhs in
            lhs.startTime > rhs.startTime
        }
    }

    private enum StreamErrorChannel {
        case motion
        case barometer
        case location
    }

    private func advanceStreamGeneration() -> Int {
        streamGenerationLock.lock()
        streamGeneration += 1
        let generation = streamGeneration
        streamGenerationLock.unlock()
        return generation
    }

    private func currentStreamGeneration() -> Int {
        streamGenerationLock.lock()
        let generation = streamGeneration
        streamGenerationLock.unlock()
        return generation
    }

    private func isCurrentGeneration(_ generation: Int) -> Bool {
        streamGenerationLock.lock()
        let isCurrent = generation == streamGeneration
        streamGenerationLock.unlock()
        return isCurrent
    }

    private func enqueueFusion(
        generation: Int,
        dropsWhenBacklogged: Bool = false,
        _ operation: @escaping (SensorStore) -> Void
    ) {
        if dropsWhenBacklogged, fusionQueue.operationCount > maximumPendingLiveFusionOperations {
            return
        }
        fusionQueue.addOperation { [weak self] in
            guard let self, self.isCurrentGeneration(generation) else { return }
            operation(self)
        }
    }

    private func startStreamHealthMonitor(generation: Int) {
        streamHealthTask?.cancel()
        streamHealthTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 500_000_000)
                guard let self else { return }
                await MainActor.run {
                    guard self.isCurrentGeneration(generation) else { return }
                    self.updateStreamHealth(now: Date(), force: true)
                }
            }
        }
    }

    private func updateStreamHealth(now: Date, force: Bool = false) {
        let uptime = ProcessInfo.processInfo.systemUptime
        if !force,
           let lastStreamHealthPublishTS,
           uptime - lastStreamHealthPublishTS < streamHealthPublishMinDtSec {
            return
        }
        lastStreamHealthPublishTS = uptime
        streamHealth = StreamHealth.evaluate(
            now: now,
            lastImuSampleDate: lastImuUiSampleDate,
            lastGnssSampleDate: locationTimestamp,
            motionError: lastMotionErrorMessage,
            locationError: lastLocationErrorMessage,
            recordingError: lastRecordingErrorMessage,
            isRecording: isRecording
        )
        if streamHealth.imu == .stale {
            fusionConfidence = min(fusionConfidence, 0.2)
        }
    }

    private func publishStreamError(channel: StreamErrorChannel, error: Error) {
        publishStreamError(channel: channel, message: error.localizedDescription)
    }

    private func publishStreamError(channel: StreamErrorChannel, message: String) {
        Task { @MainActor in
            switch channel {
            case .motion:
                self.lastMotionErrorMessage = message
            case .barometer:
                break
            case .location:
                self.lastLocationErrorMessage = message
            }
            self.updateStreamHealth(now: Date(), force: true)
        }
    }

    private func isUsableLiveLocation(_ location: CLLocation, now: Date = Date()) -> Bool {
        guard CLLocationCoordinate2DIsValid(location.coordinate) else { return false }
        guard location.horizontalAccuracy >= 0.0 else { return false }
        guard now.timeIntervalSince(location.timestamp) <= 5.0 else { return false }
        return true
    }
}

extension SensorStore: CLLocationManagerDelegate {
    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        Task { @MainActor in
            self.authorization = manager.authorizationStatus
            if self.authorization == .authorizedWhenInUse {
                manager.requestAlwaysAuthorization()
            }
            if self.authorization == .authorizedWhenInUse || self.authorization == .authorizedAlways {
                manager.startUpdatingLocation()
                manager.startUpdatingHeading()
            }
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let loc = locations.last else { return }
        guard streamMode == .playback || isUsableLiveLocation(loc) else {
            publishStreamError(channel: .location, message: "Location sample was stale or invalid.")
            return
        }
        lastLocationErrorMessage = nil
        let generation = currentStreamGeneration()
        let nedPosition = computeNEDPositionFromGnss(current: loc)
        let nedVelocity = computeNEDVelocityFromGnss(current: loc)

        let posN = nedPosition.n
        let posE = nedPosition.e
        let posD = nedPosition.d
        let velN = nedVelocity.n
        let velE = nedVelocity.e
        let velD = self.velDownMps
        let hAcc = loc.horizontalAccuracy
        let vAcc = loc.verticalAccuracy
        let latitudeDeg = loc.coordinate.latitude
        let longitudeDeg = loc.coordinate.longitude
        let altitudeM = loc.altitude
        let courseDeg = loc.course >= 0 ? loc.course : nil
        let speedMps = loc.speed >= 0 ? loc.speed : nil
        let speedAccuracyMps = loc.speedAccuracy >= 0 ? loc.speedAccuracy : nil
        let courseAccuracyDeg = loc.courseAccuracy >= 0 ? loc.courseAccuracy : nil
        recordGnssSample(
            sampleDate: loc.timestamp,
            latitudeDeg: latitudeDeg,
            longitudeDeg: longitudeDeg,
            altitudeM: altitudeM,
            horizontalAccuracyM: hAcc >= 0 ? hAcc : nil,
            verticalAccuracyM: vAcc >= 0 ? vAcc : nil,
            speedMps: speedMps,
            courseDeg: courseDeg,
            speedAccuracyMps: speedAccuracyMps,
            courseAccuracyDeg: courseAccuracyDeg,
            posN: posN,
            posE: posE,
            posD: posD,
            velN: velN,
            velE: velE,
            velD: velD
        )
        enqueueFusion(generation: generation) { store in
            store.runEkfFuseGps(
                sampleDate: loc.timestamp,
                latitudeDeg: latitudeDeg,
                longitudeDeg: longitudeDeg,
                altitudeM: altitudeM,
                posN: posN,
                posE: posE,
                posD: posD,
                velN: velN,
                velE: velE,
                velD: velD,
                hAcc: hAcc,
                vAcc: vAcc,
                courseDeg: courseDeg,
                speedAccuracyMps: speedAccuracyMps
            )
        }

        Task { @MainActor in
            guard self.isCurrentGeneration(generation) else { return }
            self.publishGnssSample(
                sampleDate: loc.timestamp,
                latitudeDeg: latitudeDeg,
                longitudeDeg: longitudeDeg,
                altitudeM: altitudeM,
                posN: nedPosition.n,
                posE: nedPosition.e,
                posD: nedPosition.d,
                speedMps: speedMps,
                courseDeg: courseDeg,
                velN: nedVelocity.n,
                velE: nedVelocity.e,
                velD: self.velDownMps,
                horizontalAccuracyM: hAcc >= 0 ? hAcc : nil,
                verticalAccuracyM: vAcc >= 0 ? vAcc : nil
            )
        }
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("Location error: \(error)")
        publishStreamError(channel: .location, error: error)
    }

    private func computeNEDVelocityFromGnss(current: CLLocation) -> (n: Double?, e: Double?, d: Double?) {
        guard let velocity = GnssVelocityResolver.horizontalVelocity(
            speedMps: current.speed,
            courseDeg: current.course
        ) else { return (nil, nil, nil) }
        // Vertical velocity is provided by barometer updates in this app.
        return (velocity.northMps, velocity.eastMps, nil)
    }

    private func computeNEDPositionFromGnss(current: CLLocation) -> (n: Double?, e: Double?, d: Double?) {
        if nedReference == nil {
            nedReference = current
            return (0.0, 0.0, 0.0)
        }

        guard let reference = nedReference else {
            return (nil, nil, nil)
        }

        let lat0Rad = reference.coordinate.latitude * .pi / 180.0
        let metersPerDegLat = 111_132.92
            - 559.82 * cos(2.0 * lat0Rad)
            + 1.175 * cos(4.0 * lat0Rad)
            - 0.0023 * cos(6.0 * lat0Rad)
        let metersPerDegLon = 111_412.84 * cos(lat0Rad)
            - 93.5 * cos(3.0 * lat0Rad)
            + 0.118 * cos(5.0 * lat0Rad)

        let dLatDeg = current.coordinate.latitude - reference.coordinate.latitude
        let dLonDeg = current.coordinate.longitude - reference.coordinate.longitude
        let dAltM = current.altitude - reference.altitude

        let n = dLatDeg * metersPerDegLat
        let e = dLonDeg * metersPerDegLon
        let d = -dAltM
        return (n, e, d)
    }

    private func runEkfPredict(
        sampleDate: Date,
        ax: Double,
        ay: Double,
        az: Double,
        gx: Double,
        gy: Double,
        gz: Double
    ) {
        let processingDate = FusionTimingPolicy.processingDate(
            for: sampleDate,
            after: lastFusionSampleDate
        )
        let result = fusionEngine.processImu(
            sampleDate: processingDate,
            accelMps2: (x: ax, y: ay, z: az),
            gyroRadps: (x: gx, y: gy, z: gz)
        )
        lastFusionSampleDate = processingDate
        lastFusionImuSampleDate = sampleDate
        publishFusionResult(result, sampleDate: sampleDate)
    }

    private func runEkfFuseGps(
        sampleDate: Date,
        latitudeDeg: Double,
        longitudeDeg: Double,
        altitudeM: Double,
        posN: Double?,
        posE: Double?,
        posD: Double?,
        velN: Double?,
        velE: Double?,
        velD: Double?,
        hAcc: Double,
        vAcc: Double,
        courseDeg: Double?,
        speedAccuracyMps: Double?
    ) {
        _ = posN
        _ = posE
        _ = posD
        guard FusionTimingPolicy.hasFreshImu(
            for: sampleDate,
            lastImuSampleDate: lastFusionImuSampleDate
        ) else {
            return
        }
        guard let input = GnssFusionInput.make(
            latitudeDeg: latitudeDeg,
            longitudeDeg: longitudeDeg,
            altitudeM: altitudeM,
            velN: velN,
            velE: velE,
            velD: velD,
            hAcc: hAcc,
            vAcc: vAcc,
            courseDeg: courseDeg,
            speedAccuracyMps: speedAccuracyMps
        ) else {
            return
        }
        let processingDate = FusionTimingPolicy.processingDate(
            for: sampleDate,
            after: lastFusionSampleDate
        )
        let result = fusionEngine.processGnss(
            sampleDate: processingDate,
            latitudeDeg: latitudeDeg,
            longitudeDeg: longitudeDeg,
            altitudeM: altitudeM,
            positionStdM: (
                n: input.positionStdM.north,
                e: input.positionStdM.east,
                d: input.positionStdM.down
            ),
            velocityNedMps: (
                n: input.velocityNedMps.north,
                e: input.velocityNedMps.east,
                d: input.velocityNedMps.down
            ),
            velocityStdMps: (
                n: input.velocityStdMps.north,
                e: input.velocityStdMps.east,
                d: input.velocityStdMps.down
            ),
            headingRad: input.headingRad
        )
        lastFusionSampleDate = processingDate
        publishFusionResult(result, sampleDate: sampleDate)
    }

    private func resetFusionEngineState() {
        lastFusionSampleDate = nil
        lastFusionImuSampleDate = nil
        fusionEngine.resetReducedAuto()
    }

    private func publishFusionResult(_ result: FusionResult?, sampleDate: Date) {
        guard let result else { return }
        guard shouldPublishFusionUi(status: result.status) else { return }
        Task { @MainActor in
            let tSec = self.relativeTimeSeconds(for: sampleDate)
            self.applyFusionStatus(result.status)
            if let snapshot = result.snapshot {
                self.appendEkfSamplesFromState(snapshot, tSec: tSec)
            }
        }
    }

    private func shouldPublishFusionUi(status: FusionStatus) -> Bool {
        let now = ProcessInfo.processInfo.systemUptime
        let isStateTransition = status.mountReadyChanged
            || status.reducedInitializedNow
            || status.filterInitializedNow
        if isStateTransition {
            lastFusionUiPublishTS = now
            return true
        }
        guard let lastFusionUiPublishTS else {
            self.lastFusionUiPublishTS = now
            return true
        }
        guard now - lastFusionUiPublishTS >= fusionUiPublishMinDtSec else {
            return false
        }
        self.lastFusionUiPublishTS = now
        return true
    }

    @MainActor
    private func applyFusionStatus(_ status: FusionStatus) {
        ekfInitialized = status.filterInitialized
        ekfMountReady = status.mountReady
        let health = FusionHealth.evaluate(
            mountReady: status.mountReady,
            initialized: status.filterInitialized,
            gnssAccuracy: GnssAccuracy(
                horizontalAccuracyM: horizontalAccuracyM,
                verticalAccuracyM: verticalAccuracyM
            ),
            streamHealth: streamHealth
        )
        fusionConfidence = health.fusedConfidence
    }

    @MainActor
    private func appendEkfSamplesFromState(_ snapshot: FusionSnapshot, tSec: Double) {
        ekfInitialized = snapshot.initialized
        ekfMountReady = snapshot.mountReady
        fusedPosNorthM = snapshot.positionNedM.north
        fusedPosEastM = snapshot.positionNedM.east
        fusedPosDownM = snapshot.positionNedM.down
        if let coordinate = snapshot.coordinate {
            fusedLatitude = coordinate.latitudeDeg
            fusedLongitude = coordinate.longitudeDeg
            fusedAltitudeM = coordinate.altitudeM
        }

        let nedVelocity = NavigationVectorNED(
            north: snapshot.velocityNedMps.n,
            east: snapshot.velocityNedMps.e,
            down: snapshot.velocityNedMps.d
        )
        let health = FusionHealth.evaluate(
            mountReady: snapshot.mountReady,
            initialized: snapshot.initialized,
            gnssAccuracy: GnssAccuracy(
                horizontalAccuracyM: horizontalAccuracyM,
                verticalAccuracyM: verticalAccuracyM
            ),
            streamHealth: streamHealth
        )
        let motionDisplay = VehicleMotionDisplay.make(
            nedVelocityMps: nedVelocity,
            attitudeQNV: snapshot.attitudeQNV,
            yawRateRadps: motion.gz,
            longitudinalAccelerationMps2: motion.ax,
            verticalAccelerationMps2: motion.az,
            health: health
        )
        vehicleForwardMps = motionDisplay.vehicleVelocityFRDMps.forward
        vehicleRightMps = motionDisplay.vehicleVelocityFRDMps.right
        vehicleDownMps = motionDisplay.vehicleVelocityFRDMps.down
        vehicleSegment = motionDisplay.segment
        fusionConfidence = health.fusedConfidence

        appendSample(
            to: &fusedPositionHistory,
            sample: TimedVec3Sample(
                tSec: tSec,
                x: snapshot.positionNedM.north,
                y: snapshot.positionNedM.east,
                z: snapshot.positionNedM.down
            ),
            maxCount: chartHistoryMaxCount
        )
        appendRouteSample(
            to: &fusedRouteHistory,
            sample: TimedVec3Sample(
                tSec: tSec,
                x: snapshot.positionNedM.north,
                y: snapshot.positionNedM.east,
                z: snapshot.positionNedM.down
            ),
            minIntervalSec: routeSampleMinDtSec
        )
        appendSample(
            to: &ekfVelocityHistory,
            sample: TimedVec3Sample(
                tSec: tSec,
                x: snapshot.velocityNedMps.n,
                y: snapshot.velocityNedMps.e,
                z: snapshot.velocityNedMps.d
            ),
            maxCount: chartHistoryMaxCount
        )
        appendSample(
            to: &ekfEulerHistory,
            sample: TimedVec3Sample(
                tSec: tSec,
                x: snapshot.eulerRad.roll * 180.0 / .pi,
                y: snapshot.eulerRad.pitch * 180.0 / .pi,
                z: snapshot.eulerRad.yaw * 180.0 / .pi
            ),
            maxCount: chartHistoryMaxCount
        )
        appendSample(
            to: &ekfGyroBiasHistory,
            sample: TimedVec3Sample(
                tSec: tSec,
                x: snapshot.gyroBiasRadps.x,
                y: snapshot.gyroBiasRadps.y,
                z: snapshot.gyroBiasRadps.z
            ),
            maxCount: chartHistoryMaxCount
        )
        appendSample(
            to: &ekfAccelBiasHistory,
            sample: TimedVec3Sample(
                tSec: tSec,
                x: snapshot.accelBiasMps2.x,
                y: snapshot.accelBiasMps2.y,
                z: snapshot.accelBiasMps2.z
            ),
            maxCount: chartHistoryMaxCount
        )
    }

    @MainActor
    private func publishImuSample(
        ax: Double,
        ay: Double,
        az: Double,
        gx: Double,
        gy: Double,
        gz: Double,
        sampleDate: Date
    ) {
        lastImuUiSampleDate = sampleDate
        lastMotionErrorMessage = nil
        motion = MotionSample(
            ax: ax,
            ay: ay,
            az: az,
            gx: gx,
            gy: gy,
            gz: gz,
            timestamp: sampleDate
        )
        let tSec = relativeTimeSeconds(for: sampleDate)
        appendSample(
            to: &imuAccelHistory,
            sample: TimedVec3Sample(
                tSec: tSec,
                x: ax,
                y: ay,
                z: az
            ),
            maxCount: chartHistoryMaxCount
        )
        appendSample(
            to: &imuGyroHistory,
            sample: TimedVec3Sample(
                tSec: tSec,
                x: gx,
                y: gy,
                z: gz
            ),
            maxCount: chartHistoryMaxCount
        )
        updateStreamHealth(now: Date(), force: false)
    }

    @MainActor
    private func publishGnssSample(
        sampleDate: Date,
        latitudeDeg: Double,
        longitudeDeg: Double,
        altitudeM: Double,
        posN: Double?,
        posE: Double?,
        posD: Double?,
        speedMps: Double?,
        courseDeg: Double?,
        velN: Double?,
        velE: Double?,
        velD: Double?,
        horizontalAccuracyM: Double?,
        verticalAccuracyM: Double?
    ) {
        lastLocationErrorMessage = nil
        latitude = latitudeDeg
        longitude = longitudeDeg
        self.altitudeM = altitudeM
        posNorthM = posN
        posEastM = posE
        posDownM = posD
        self.speedMps = speedMps
        self.courseDeg = courseDeg
        velNorthMps = velN
        velEastMps = velE
        if let velD {
            velDownMps = velD
        }
        self.horizontalAccuracyM = horizontalAccuracyM
        self.verticalAccuracyM = verticalAccuracyM
        locationTimestamp = sampleDate

        let tSec = relativeTimeSeconds(for: sampleDate)
        appendSample(
            to: &nedPositionHistory,
            sample: TimedVec3Sample(
                tSec: tSec,
                x: posN,
                y: posE,
                z: posD
            ),
            maxCount: chartHistoryMaxCount
        )
        appendRouteSample(
            to: &gnssRouteHistory,
            sample: TimedVec3Sample(
                tSec: tSec,
                x: posN,
                y: posE,
                z: posD
            )
        )
        appendSample(
            to: &nedVelocityHistory,
            sample: TimedVec3Sample(
                tSec: tSec,
                x: velN,
                y: velE,
                z: velDownMps
            ),
            maxCount: chartHistoryMaxCount
        )
        updateStreamHealth(now: Date(), force: false)
    }

    @MainActor
    private func publishBarometerVelocity(vDown: Double, sampleDate: Date) {
        velDownMps = vDown
        let tSec = relativeTimeSeconds(for: sampleDate)
        appendSample(
            to: &nedVelocityHistory,
            sample: TimedVec3Sample(
                tSec: tSec,
                x: velNorthMps,
                y: velEastMps,
                z: velDownMps
            ),
            maxCount: chartHistoryMaxCount
        )
    }

#if DEBUG
    @MainActor
    private func publishIosAttitude(
        rollRad: Double,
        pitchRad: Double,
        yawRad: Double,
        sampleDate: Date
    ) {
        iosAttitudeEulerDeg = TimedVec3Sample(
            tSec: relativeTimeSeconds(for: sampleDate),
            x: rollRad * 180.0 / .pi,
            y: pitchRad * 180.0 / .pi,
            z: yawRad * 180.0 / .pi
        )
    }
#endif

    private func dateFromMotionTimestamp(_ timestampSec: TimeInterval) -> Date {
        let uptimeDelta = timestampSec - ProcessInfo.processInfo.systemUptime
        return Date(timeIntervalSinceNow: uptimeDelta)
    }

    private func bodySpecificForceMps2(from data: CMDeviceMotion) -> (x: Double, y: Double, z: Double) {
        // Treated as raw body-frame specific force for now; validate axes/signs on-device before trusting EKF traces.
        (
            x: (data.userAcceleration.x + data.gravity.x) * g0Mps2,
            y: (data.userAcceleration.y + data.gravity.y) * g0Mps2,
            z: (data.userAcceleration.z + data.gravity.z) * g0Mps2
        )
    }

    private func relativeTimeSeconds(for date: Date) -> Double {
        if streamStartTime == nil {
            streamStartTime = date
        }
        guard let start = streamStartTime else { return 0.0 }
        return date.timeIntervalSince(start)
    }

    private func appendSample(
        to buffer: inout [TimedVec3Sample],
        sample: TimedVec3Sample,
        maxCount: Int
    ) {
        buffer.append(sample)
        if buffer.count > maxCount {
            buffer.removeFirst(buffer.count - maxCount)
        }
    }

    private func appendRouteSample(
        to buffer: inout [TimedVec3Sample],
        sample: TimedVec3Sample,
        minIntervalSec: Double? = nil
    ) {
        if let minIntervalSec,
           let previous = buffer.last,
           sample.tSec - previous.tSec < minIntervalSec {
            return
        }
        appendSample(to: &buffer, sample: sample, maxCount: routeHistoryMaxCount)
    }
}
