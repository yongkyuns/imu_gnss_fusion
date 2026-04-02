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
    @Published var nedPositionHistory: [TimedVec3Sample] = []
    @Published var nedVelocityHistory: [TimedVec3Sample] = []
    @Published var imuAccelHistory: [TimedVec3Sample] = []
    @Published var imuGyroHistory: [TimedVec3Sample] = []
    @Published var ekfVelocityHistory: [TimedVec3Sample] = []
    @Published var ekfEulerHistory: [TimedVec3Sample] = []
    @Published var ekfGyroBiasHistory: [TimedVec3Sample] = []
    @Published var ekfAccelBiasHistory: [TimedVec3Sample] = []

    private let locationManager = CLLocationManager()
    private let motionManager = CMMotionManager()
    private let altimeter = CMAltimeter()
    private let motionQueue = OperationQueue()
    private let barometerQueue = OperationQueue()
    private var nedReference: CLLocation?
    private var lastBarometerSample: (hM: Double, tS: TimeInterval)?
    private var filteredVerticalUpMps: Double?
    private var streamStartTime: Date?
    private var lastMotionPublishTS: TimeInterval?
    private var lastBaroPublishTS: TimeInterval?
    private let motionPublishMinDtSec = 1.0 / 20.0
    private let baroPublishMinDtSec = 1.0 / 10.0
    override init() {
        super.init()
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
        locationManager.distanceFilter = kCLDistanceFilterNone
        locationManager.activityType = .otherNavigation
        authorization = locationManager.authorizationStatus

        motionQueue.name = "imu_gnss_phone.motion.queue"
        motionQueue.qualityOfService = .userInitiated
        motionQueue.maxConcurrentOperationCount = 1

        barometerQueue.name = "imu_gnss_phone.barometer.queue"
        barometerQueue.qualityOfService = .userInitiated
        barometerQueue.maxConcurrentOperationCount = 1

    }

    func start() {
        nedReference = nil
        lastBarometerSample = nil
        filteredVerticalUpMps = nil
        streamStartTime = nil
        lastMotionPublishTS = nil
        lastBaroPublishTS = nil
        nedPositionHistory.removeAll(keepingCapacity: true)
        nedVelocityHistory.removeAll(keepingCapacity: true)
        imuAccelHistory.removeAll(keepingCapacity: true)
        imuGyroHistory.removeAll(keepingCapacity: true)
        ekfVelocityHistory.removeAll(keepingCapacity: true)
        ekfEulerHistory.removeAll(keepingCapacity: true)
        ekfGyroBiasHistory.removeAll(keepingCapacity: true)
        ekfAccelBiasHistory.removeAll(keepingCapacity: true)

        if authorization == .notDetermined {
            locationManager.requestWhenInUseAuthorization()
        }
        if authorization == .authorizedWhenInUse || authorization == .authorizedAlways {
            locationManager.startUpdatingLocation()
            locationManager.startUpdatingHeading()
        }

        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 1.0 / 100.0
            motionManager.startDeviceMotionUpdates(to: motionQueue) { [weak self] data, _ in
                guard let self, let data else { return }
                let timestampSec = data.timestamp
                let ax = (data.userAcceleration.x + data.gravity.x) * g0Mps2
                let ay = (data.userAcceleration.y + data.gravity.y) * g0Mps2
                let az = (data.userAcceleration.z + data.gravity.z) * g0Mps2
                let gx = data.rotationRate.x
                let gy = data.rotationRate.y
                let gz = data.rotationRate.z

                self.runEkfPredict(
                    timestampSec: timestampSec,
                    ax: ax,
                    ay: ay,
                    az: az,
                    gx: gx,
                    gy: gy,
                    gz: gz
                )

                if let last = self.lastMotionPublishTS, (data.timestamp - last) < self.motionPublishMinDtSec {
                    return
                }
                self.lastMotionPublishTS = data.timestamp
                Task { @MainActor in
                    self.motion = MotionSample(
                        ax: ax,
                        ay: ay,
                        az: az,
                        gx: gx,
                        gy: gy,
                        gz: gz,
                        timestamp: Date()
                    )
                    let tSec = self.relativeTimeSeconds(for: self.motion.timestamp)
                    self.appendSample(
                        to: &self.imuAccelHistory,
                        sample: TimedVec3Sample(
                            tSec: tSec,
                            x: self.motion.ax,
                            y: self.motion.ay,
                            z: self.motion.az
                        ),
                        maxCount: 240
                    )
                    self.appendSample(
                        to: &self.imuGyroHistory,
                        sample: TimedVec3Sample(
                            tSec: tSec,
                            x: self.motion.gx,
                            y: self.motion.gy,
                            z: self.motion.gz
                        ),
                        maxCount: 240
                    )
                    self.appendEkfSamplesFromState(tSec: tSec)
                }
            }
        }

        if CMAltimeter.isRelativeAltitudeAvailable() {
            altimeter.startRelativeAltitudeUpdates(to: barometerQueue) { [weak self] data, _ in
                guard let self, let data else { return }
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

                guard let vDown else { return }
                if let last = self.lastBaroPublishTS, (tS - last) < self.baroPublishMinDtSec {
                    return
                }
                self.lastBaroPublishTS = tS
                Task { @MainActor in
                    self.velDownMps = vDown
                    let now = Date()
                    let tSec = self.relativeTimeSeconds(for: now)
                    self.appendSample(
                        to: &self.nedVelocityHistory,
                        sample: TimedVec3Sample(
                            tSec: tSec,
                            x: self.velNorthMps,
                            y: self.velEastMps,
                            z: self.velDownMps
                        ),
                        maxCount: 240
                    )
                }
            }
        } else {
            Task { @MainActor in
                self.velDownMps = nil
            }
        }
    }

    func stop() {
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
    }
}

extension SensorStore: CLLocationManagerDelegate {
    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        Task { @MainActor in
            self.authorization = manager.authorizationStatus
            if self.authorization == .authorizedWhenInUse || self.authorization == .authorizedAlways {
                manager.startUpdatingLocation()
                manager.startUpdatingHeading()
            }
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let loc = locations.last else { return }
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
        let tSecForEkf = loc.timestamp.timeIntervalSince1970
        motionQueue.addOperation { [weak self] in
            self?.runEkfFuseGps(
                posN: posN,
                posE: posE,
                posD: posD,
                velN: velN,
                velE: velE,
                velD: velD,
                hAcc: hAcc,
                vAcc: vAcc,
                timestampSec: tSecForEkf
            )
        }

        Task { @MainActor in
            self.latitude = loc.coordinate.latitude
            self.longitude = loc.coordinate.longitude
            self.altitudeM = loc.altitude
            self.posNorthM = nedPosition.n
            self.posEastM = nedPosition.e
            self.posDownM = nedPosition.d
            self.speedMps = loc.speed >= 0 ? loc.speed : nil
            self.courseDeg = loc.course >= 0 ? loc.course : nil
            self.velNorthMps = nedVelocity.n
            self.velEastMps = nedVelocity.e
            self.horizontalAccuracyM = loc.horizontalAccuracy
            self.verticalAccuracyM = loc.verticalAccuracy
            self.locationTimestamp = loc.timestamp

            let tSec = self.relativeTimeSeconds(for: loc.timestamp)
            self.appendSample(
                to: &self.nedPositionHistory,
                sample: TimedVec3Sample(
                    tSec: tSec,
                    x: self.posNorthM,
                    y: self.posEastM,
                    z: self.posDownM
                ),
                maxCount: 240
            )
            self.appendSample(
                to: &self.nedVelocityHistory,
                sample: TimedVec3Sample(
                    tSec: tSec,
                    x: self.velNorthMps,
                    y: self.velEastMps,
                    z: self.velDownMps
                ),
                maxCount: 240
            )
            self.appendEkfSamplesFromState(tSec: tSec)
        }
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("Location error: \(error)")
    }

    private func computeNEDVelocityFromGnss(current: CLLocation) -> (n: Double?, e: Double?, d: Double?) {
        guard current.speed >= 0, current.course >= 0 else {
            return (nil, nil, nil)
        }
        let headingRad = current.course * .pi / 180.0
        let vn = current.speed * cos(headingRad)
        let ve = current.speed * sin(headingRad)
        // Vertical velocity is provided by barometer updates in this app.
        return (vn, ve, nil)
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
        timestampSec: TimeInterval,
        ax: Double,
        ay: Double,
        az: Double,
        gx: Double,
        gy: Double,
        gz: Double
    ) {
        _ = timestampSec
        _ = ax
        _ = ay
        _ = az
        _ = gx
        _ = gy
        _ = gz
    }

    private func runEkfFuseGps(
        posN: Double?,
        posE: Double?,
        posD: Double?,
        velN: Double?,
        velE: Double?,
        velD: Double?,
        hAcc: Double,
        vAcc: Double,
        timestampSec: TimeInterval
    ) {
        _ = posN
        _ = posE
        _ = posD
        _ = velN
        _ = velE
        _ = velD
        _ = hAcc
        _ = vAcc
        _ = timestampSec
    }

    @MainActor
    private func appendEkfSamplesFromState(tSec: Double) {
        _ = tSec
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
}
