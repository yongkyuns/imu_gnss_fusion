//! C ABI wrapper for the Rust `sensor_fusion` facade.

use core::ptr;

use sensor_fusion::{GnssSample, ImuSample, SensorFusion, Update};

/// Opaque fusion handle owned by Rust and passed across the C ABI as a pointer.
pub struct SensorFusionFfi {
    inner: SensorFusion,
    last_update: SensorFusionFfiUpdate,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SensorFusionFfiUpdate {
    pub mount_ready: bool,
    pub mount_ready_changed: bool,
    pub ekf_initialized: bool,
    pub ekf_initialized_now: bool,
    pub filter_initialized: bool,
    pub filter_initialized_now: bool,
    pub mount_q_bv_valid: bool,
    pub mount_q_bv: [f32; 4],
}

impl Default for SensorFusionFfiUpdate {
    fn default() -> Self {
        Self {
            mount_ready: false,
            mount_ready_changed: false,
            ekf_initialized: false,
            ekf_initialized_now: false,
            filter_initialized: false,
            filter_initialized_now: false,
            mount_q_bv_valid: false,
            mount_q_bv: [1.0, 0.0, 0.0, 0.0],
        }
    }
}

impl From<Update> for SensorFusionFfiUpdate {
    fn from(update: Update) -> Self {
        Self {
            mount_ready: update.mount_ready,
            mount_ready_changed: update.mount_ready_changed,
            ekf_initialized: update.ekf_initialized,
            ekf_initialized_now: update.ekf_initialized_now,
            filter_initialized: update.ekf_initialized,
            filter_initialized_now: update.ekf_initialized_now,
            mount_q_bv_valid: update.mount_q_bv.is_some(),
            mount_q_bv: update.mount_q_bv.unwrap_or([1.0, 0.0, 0.0, 0.0]),
        }
    }
}

impl SensorFusionFfiUpdate {
    fn from_fusion_state(fusion: &SensorFusion) -> Self {
        let filter_initialized = fusion.ekf().is_some();
        let mount_q_bv = fusion.mount_q_bv();

        Self {
            mount_ready: fusion.mount_ready(),
            ekf_initialized: fusion.ekf().is_some(),
            filter_initialized,
            mount_q_bv_valid: mount_q_bv.is_some(),
            mount_q_bv: mount_q_bv.unwrap_or([1.0, 0.0, 0.0, 0.0]),
            ..Self::default()
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SensorFusionFfiEkfSnapshot {
    pub mount_ready: bool,
    pub initialized: bool,
    pub q0: f32,
    pub q1: f32,
    pub q2: f32,
    pub q3: f32,
    pub vel_n_mps: f32,
    pub vel_e_mps: f32,
    pub vel_d_mps: f32,
    pub pos_n_m: f32,
    pub pos_e_m: f32,
    pub pos_d_m: f32,
    pub gyro_bias_x_radps: f32,
    pub gyro_bias_y_radps: f32,
    pub gyro_bias_z_radps: f32,
    pub accel_bias_x_mps2: f32,
    pub accel_bias_y_mps2: f32,
    pub accel_bias_z_mps2: f32,
    pub q_bv0: f32,
    pub q_bv1: f32,
    pub q_bv2: f32,
    pub q_bv3: f32,
    pub position_lla_valid: bool,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub height_m: f64,
}

impl Default for SensorFusionFfiEkfSnapshot {
    fn default() -> Self {
        Self {
            mount_ready: false,
            initialized: false,
            q0: 1.0,
            q1: 0.0,
            q2: 0.0,
            q3: 0.0,
            vel_n_mps: 0.0,
            vel_e_mps: 0.0,
            vel_d_mps: 0.0,
            pos_n_m: 0.0,
            pos_e_m: 0.0,
            pos_d_m: 0.0,
            gyro_bias_x_radps: 0.0,
            gyro_bias_y_radps: 0.0,
            gyro_bias_z_radps: 0.0,
            accel_bias_x_mps2: 0.0,
            accel_bias_y_mps2: 0.0,
            accel_bias_z_mps2: 0.0,
            q_bv0: 1.0,
            q_bv1: 0.0,
            q_bv2: 0.0,
            q_bv3: 0.0,
            position_lla_valid: false,
            lat_deg: 0.0,
            lon_deg: 0.0,
            height_m: 0.0,
        }
    }
}

impl SensorFusionFfi {
    fn new(inner: SensorFusion) -> Self {
        let last_update = SensorFusionFfiUpdate::from_fusion_state(&inner);
        Self { inner, last_update }
    }

    fn status(&self) -> SensorFusionFfiUpdate {
        let mut status = SensorFusionFfiUpdate::from_fusion_state(&self.inner);
        status.mount_ready_changed = self.last_update.mount_ready_changed;
        status.ekf_initialized_now = self.last_update.ekf_initialized_now;
        status.filter_initialized_now = self.last_update.filter_initialized_now;
        status
    }

    fn reset(&mut self, inner: SensorFusion) {
        self.inner = inner;
        self.last_update = SensorFusionFfiUpdate::from_fusion_state(&self.inner);
    }

    fn store_update(&mut self, update: Update) -> SensorFusionFfiUpdate {
        self.last_update = update.into();
        self.status()
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sensor_fusion_create_ekf_auto() -> *mut SensorFusionFfi {
    Box::into_raw(Box::new(SensorFusionFfi::new(SensorFusion::new())))
}

#[unsafe(no_mangle)]
pub extern "C" fn sensor_fusion_create_ekf_manual(
    qw: f32,
    qx: f32,
    qy: f32,
    qz: f32,
) -> *mut SensorFusionFfi {
    Box::into_raw(Box::new(SensorFusionFfi::new(SensorFusion::with_mount([
        qw, qx, qy, qz,
    ]))))
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `handle` must be either null or a pointer returned by this crate's create
/// functions that has not already been destroyed.
pub unsafe extern "C" fn sensor_fusion_destroy(handle: *mut SensorFusionFfi) {
    if handle.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(handle));
    }
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `handle` must be either null or a valid pointer returned by this crate's
/// create functions.
pub unsafe extern "C" fn sensor_fusion_reset_ekf_auto(handle: *mut SensorFusionFfi) {
    let Some(fusion) = fusion_mut(handle) else {
        return;
    };
    fusion.reset(SensorFusion::new());
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `handle` must be either null or a valid pointer returned by this crate's
/// create functions.
pub unsafe extern "C" fn sensor_fusion_reset_ekf_manual(
    handle: *mut SensorFusionFfi,
    qw: f32,
    qx: f32,
    qy: f32,
    qz: f32,
) {
    let Some(fusion) = fusion_mut(handle) else {
        return;
    };
    fusion.reset(SensorFusion::with_mount([qw, qx, qy, qz]));
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `handle` must be either null or a valid pointer returned by this crate's
/// create functions.
pub unsafe extern "C" fn sensor_fusion_snapshot_status(
    handle: *const SensorFusionFfi,
) -> SensorFusionFfiUpdate {
    let Some(fusion) = fusion_ref(handle) else {
        return SensorFusionFfiUpdate::default();
    };
    fusion.status()
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `handle` must be either null or a valid pointer returned by this crate's
/// create functions.
pub unsafe extern "C" fn sensor_fusion_process_imu(
    handle: *mut SensorFusionFfi,
    t_s: f32,
    ax: f32,
    ay: f32,
    az: f32,
    gx: f32,
    gy: f32,
    gz: f32,
) -> SensorFusionFfiUpdate {
    let Some(fusion) = fusion_mut(handle) else {
        return SensorFusionFfiUpdate::default();
    };

    let update = fusion.inner.process_imu(ImuSample {
        t_s,
        gyro_radps: [gx, gy, gz],
        accel_mps2: [ax, ay, az],
    });
    fusion.store_update(update)
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
/// # Safety
///
/// `handle` must be either null or a valid pointer returned by this crate's
/// create functions.
pub unsafe extern "C" fn sensor_fusion_process_gnss(
    handle: *mut SensorFusionFfi,
    t_s: f32,
    lat_deg: f64,
    lon_deg: f64,
    height_m: f64,
    vn: f32,
    ve: f32,
    vd: f32,
    pos_std_n: f32,
    pos_std_e: f32,
    pos_std_d: f32,
    vel_std_n: f32,
    vel_std_e: f32,
    vel_std_d: f32,
    heading_rad: f32,
    is_heading_valid: bool,
) -> SensorFusionFfiUpdate {
    let Some(fusion) = fusion_mut(handle) else {
        return SensorFusionFfiUpdate::default();
    };

    let update = fusion.inner.process_gnss(GnssSample {
        t_s,
        lat_deg,
        lon_deg,
        height_m,
        vel_ned_mps: [vn, ve, vd],
        pos_std_m: [pos_std_n, pos_std_e, pos_std_d],
        vel_std_mps: [vel_std_n, vel_std_e, vel_std_d],
        heading_rad: is_heading_valid.then_some(heading_rad),
    });
    fusion.store_update(update)
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `handle` must be either null or a valid pointer returned by this crate's
/// create functions. When non-null, `out` must point to writable memory for one
/// `SensorFusionFfiEkfSnapshot`.
pub unsafe extern "C" fn sensor_fusion_snapshot_ekf(
    handle: *const SensorFusionFfi,
    out: *mut SensorFusionFfiEkfSnapshot,
) -> bool {
    if out.is_null() {
        return false;
    }

    let snapshot = ekf_snapshot(handle);
    unsafe {
        ptr::write(out, snapshot);
    }
    snapshot.initialized
}

fn fusion_mut(handle: *mut SensorFusionFfi) -> Option<&'static mut SensorFusionFfi> {
    if handle.is_null() {
        None
    } else {
        unsafe { handle.as_mut() }
    }
}

fn fusion_ref(handle: *const SensorFusionFfi) -> Option<&'static SensorFusionFfi> {
    if handle.is_null() {
        None
    } else {
        unsafe { handle.as_ref() }
    }
}

fn ekf_snapshot(handle: *const SensorFusionFfi) -> SensorFusionFfiEkfSnapshot {
    let Some(fusion) = fusion_ref(handle) else {
        return SensorFusionFfiEkfSnapshot::default();
    };

    let mut snapshot = SensorFusionFfiEkfSnapshot {
        mount_ready: fusion.inner.mount_ready(),
        ..SensorFusionFfiEkfSnapshot::default()
    };

    let Some(ekf) = fusion.inner.ekf() else {
        return snapshot;
    };

    let nominal = &ekf.nominal;
    snapshot.initialized = true;
    snapshot.q0 = nominal.q0;
    snapshot.q1 = nominal.q1;
    snapshot.q2 = nominal.q2;
    snapshot.q3 = nominal.q3;
    snapshot.vel_n_mps = nominal.vn;
    snapshot.vel_e_mps = nominal.ve;
    snapshot.vel_d_mps = nominal.vd;
    snapshot.pos_n_m = nominal.pn;
    snapshot.pos_e_m = nominal.pe;
    snapshot.pos_d_m = nominal.pd;
    snapshot.gyro_bias_x_radps = nominal.bgx;
    snapshot.gyro_bias_y_radps = nominal.bgy;
    snapshot.gyro_bias_z_radps = nominal.bgz;
    snapshot.accel_bias_x_mps2 = nominal.bax;
    snapshot.accel_bias_y_mps2 = nominal.bay;
    snapshot.accel_bias_z_mps2 = nominal.baz;
    snapshot.q_bv0 = nominal.q_bv0;
    snapshot.q_bv1 = nominal.q_bv1;
    snapshot.q_bv2 = nominal.q_bv2;
    snapshot.q_bv3 = nominal.q_bv3;
    if let Some(lla) = fusion.inner.position_lla_f64() {
        snapshot.position_lla_valid = true;
        snapshot.lat_deg = lla[0];
        snapshot.lon_deg = lla[1];
        snapshot.height_m = lla[2];
    }
    snapshot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_handles_return_defaults() {
        let update = unsafe {
            sensor_fusion_process_imu(ptr::null_mut(), 0.0, 0.0, 0.0, 9.80665, 0.0, 0.0, 0.0)
        };
        assert!(!update.filter_initialized);

        let status = unsafe { sensor_fusion_snapshot_status(ptr::null()) };
        assert!(!status.mount_ready);
        assert!(!status.ekf_initialized);
        assert!(!status.filter_initialized);
        assert!(!status.mount_q_bv_valid);
        assert_eq!(status.mount_q_bv, [1.0, 0.0, 0.0, 0.0]);

        let mut snapshot = SensorFusionFfiEkfSnapshot {
            lat_deg: 42.0,
            ..SensorFusionFfiEkfSnapshot::default()
        };
        assert!(!unsafe { sensor_fusion_snapshot_ekf(ptr::null(), &mut snapshot) });
        assert_eq!(snapshot.lat_deg, 0.0);
    }

    #[test]
    fn status_reports_pre_initialization_manual_mount_and_resets_edges() {
        let handle = sensor_fusion_create_ekf_manual(0.5, 0.5, 0.5, 0.5);
        assert!(!handle.is_null());

        let status = unsafe { sensor_fusion_snapshot_status(handle) };
        assert!(status.mount_ready);
        assert!(!status.mount_ready_changed);
        assert!(!status.ekf_initialized);
        assert!(!status.ekf_initialized_now);
        assert!(!status.filter_initialized);
        assert!(!status.filter_initialized_now);
        assert!(status.mount_q_bv_valid);
        assert_eq!(status.mount_q_bv, [0.5, 0.5, 0.5, 0.5]);

        let update = unsafe {
            sensor_fusion_process_gnss(
                handle, 1.0, 37.3318, -122.0312, 15.0, 5.0, 0.0, 0.0, 1.0, 1.0, 1.5, 0.2, 0.2, 0.2,
                0.0, true,
            )
        };
        assert!(update.ekf_initialized_now);

        let status = unsafe { sensor_fusion_snapshot_status(handle) };
        assert!(status.mount_ready);
        assert!(status.ekf_initialized);
        assert!(status.ekf_initialized_now);
        assert!(status.filter_initialized);
        assert!(status.filter_initialized_now);
        assert_eq!(status.mount_q_bv, [0.5, 0.5, 0.5, 0.5]);

        unsafe {
            sensor_fusion_reset_ekf_auto(handle);
        }
        let status = unsafe { sensor_fusion_snapshot_status(handle) };
        assert!(!status.mount_ready);
        assert!(!status.mount_ready_changed);
        assert!(!status.ekf_initialized);
        assert!(!status.ekf_initialized_now);
        assert!(!status.filter_initialized);
        assert!(!status.filter_initialized_now);
        assert!(!status.mount_q_bv_valid);
        assert_eq!(status.mount_q_bv, [1.0, 0.0, 0.0, 0.0]);

        unsafe {
            sensor_fusion_reset_ekf_manual(handle, 1.0, 0.0, 0.0, 0.0);
        }
        let status = unsafe { sensor_fusion_snapshot_status(handle) };
        assert!(status.mount_ready);
        assert!(!status.mount_ready_changed);
        assert!(!status.ekf_initialized);
        assert!(!status.ekf_initialized_now);
        assert!(!status.filter_initialized);
        assert!(!status.filter_initialized_now);
        assert!(status.mount_q_bv_valid);
        assert_eq!(status.mount_q_bv, [1.0, 0.0, 0.0, 0.0]);

        unsafe {
            sensor_fusion_destroy(handle);
        }
    }

    #[test]
    fn manual_gnss_initializes_and_snapshots_ekf_state() {
        let handle = sensor_fusion_create_ekf_manual(1.0, 0.0, 0.0, 0.0);
        assert!(!handle.is_null());

        let update = unsafe {
            sensor_fusion_process_gnss(
                handle, 1.0, 37.3318, -122.0312, 15.0, 5.0, 0.0, 0.0, 1.0, 1.0, 1.5, 0.2, 0.2, 0.2,
                0.0, true,
            )
        };
        assert!(update.mount_ready);
        assert!(update.ekf_initialized);
        assert!(update.ekf_initialized_now);

        let mut snapshot = SensorFusionFfiEkfSnapshot::default();
        assert!(unsafe { sensor_fusion_snapshot_ekf(handle, &mut snapshot) });
        assert!(snapshot.mount_ready);
        assert!(snapshot.initialized);
        assert_eq!(
            [
                snapshot.q_bv0,
                snapshot.q_bv1,
                snapshot.q_bv2,
                snapshot.q_bv3
            ],
            [1.0, 0.0, 0.0, 0.0]
        );
        assert!((snapshot.vel_n_mps - 5.0).abs() < 1.0e-6);
        assert!(snapshot.vel_e_mps.abs() < 1.0e-6);
        assert!(snapshot.vel_d_mps.abs() < 1.0e-6);
        assert!(snapshot.position_lla_valid);
        assert!((snapshot.lat_deg - 37.3318).abs() < 1.0e-6);
        assert!((snapshot.lon_deg + 122.0312).abs() < 1.0e-6);

        unsafe {
            sensor_fusion_destroy(handle);
        }
    }
}
