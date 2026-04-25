#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused_assignments)]

use crate::eskf_types::{EskfImuDelta, EskfNominalState};

pub const ERROR_STATES: usize = 18;
pub const NOISE_STATES: usize = 15;
pub const GRAVITY_MSS: f32 = 9.80665;

#[derive(Clone, Copy, Debug)]
pub struct ScalarObservation {
    pub h: [f32; ERROR_STATES],
    pub k: [f32; ERROR_STATES],
    pub s: f32,
}

pub fn predict_nominal(nominal: &mut EskfNominalState, imu: EskfImuDelta) {
    let q0 = nominal.q0;
    let q1 = nominal.q1;
    let q2 = nominal.q2;
    let q3 = nominal.q3;
    let vn = nominal.vn;
    let ve = nominal.ve;
    let vd = nominal.vd;
    let pn = nominal.pn;
    let pe = nominal.pe;
    let pd = nominal.pd;
    let bgx = nominal.bgx;
    let bgy = nominal.bgy;
    let bgz = nominal.bgz;
    let bax = nominal.bax;
    let bay = nominal.bay;
    let baz = nominal.baz;
    let dax = imu.dax;
    let day = imu.day;
    let daz = imu.daz;
    let dvx = imu.dvx;
    let dvy = imu.dvy;
    let dvz = imu.dvz;
    let dt = imu.dt;
    let g = GRAVITY_MSS;

    include!("generated_eskf/nominal_prediction_generated.rs");
}

pub fn error_transition(
    nominal: &EskfNominalState,
    imu: EskfImuDelta,
) -> (
    [[f32; ERROR_STATES]; ERROR_STATES],
    [[f32; NOISE_STATES]; ERROR_STATES],
) {
    let mut f = [[0.0; ERROR_STATES]; ERROR_STATES];
    let mut noise_input = [[0.0; NOISE_STATES]; ERROR_STATES];
    let F = &mut f;
    let G = &mut noise_input;

    let q0 = nominal.q0;
    let q1 = nominal.q1;
    let q2 = nominal.q2;
    let q3 = nominal.q3;
    let vn = nominal.vn;
    let ve = nominal.ve;
    let vd = nominal.vd;
    let pn = nominal.pn;
    let pe = nominal.pe;
    let pd = nominal.pd;
    let bgx = nominal.bgx;
    let bgy = nominal.bgy;
    let bgz = nominal.bgz;
    let bax = nominal.bax;
    let bay = nominal.bay;
    let baz = nominal.baz;
    let qcs0 = nominal.qcs0;
    let qcs1 = nominal.qcs1;
    let qcs2 = nominal.qcs2;
    let qcs3 = nominal.qcs3;
    let dax = imu.dax;
    let day = imu.day;
    let daz = imu.daz;
    let dvx = imu.dvx;
    let dvy = imu.dvy;
    let dvz = imu.dvz;
    let dt = imu.dt;
    let g = GRAVITY_MSS;

    include!("generated_eskf/error_transition_generated.rs");
    include!("generated_eskf/error_noise_input_generated.rs");

    (f, noise_input)
}

pub fn attitude_reset_jacobian(dtheta: [f32; 3]) -> [[f32; 3]; 3] {
    let mut G_reset_theta = [[0.0; 3]; 3];
    let dtheta_x = dtheta[0];
    let dtheta_y = dtheta[1];
    let dtheta_z = dtheta[2];

    include!("generated_eskf/attitude_reset_jacobian_generated.rs");

    G_reset_theta
}

pub fn gps_pos_n_observation(
    p: &[[f32; ERROR_STATES]; ERROR_STATES],
    r_pos_n: f32,
) -> ScalarObservation {
    let P = p;
    let R_POS_N = r_pos_n;
    let mut H = [0.0; ERROR_STATES];
    let mut K = [0.0; ERROR_STATES];
    let mut S = 0.0;
    include!("generated_eskf/gps_pos_n_generated.rs");
    ScalarObservation { h: H, k: K, s: S }
}

pub fn gps_pos_e_observation(
    p: &[[f32; ERROR_STATES]; ERROR_STATES],
    r_pos_e: f32,
) -> ScalarObservation {
    let P = p;
    let R_POS_E = r_pos_e;
    let mut H = [0.0; ERROR_STATES];
    let mut K = [0.0; ERROR_STATES];
    let mut S = 0.0;
    include!("generated_eskf/gps_pos_e_generated.rs");
    ScalarObservation { h: H, k: K, s: S }
}

pub fn gps_pos_d_observation(
    p: &[[f32; ERROR_STATES]; ERROR_STATES],
    r_pos_d: f32,
) -> ScalarObservation {
    let P = p;
    let R_POS_D = r_pos_d;
    let mut H = [0.0; ERROR_STATES];
    let mut K = [0.0; ERROR_STATES];
    let mut S = 0.0;
    include!("generated_eskf/gps_pos_d_generated.rs");
    ScalarObservation { h: H, k: K, s: S }
}

pub fn gps_vel_n_observation(
    p: &[[f32; ERROR_STATES]; ERROR_STATES],
    r_vel_n: f32,
) -> ScalarObservation {
    let P = p;
    let R_VEL_N = r_vel_n;
    let mut H = [0.0; ERROR_STATES];
    let mut K = [0.0; ERROR_STATES];
    let mut S = 0.0;
    include!("generated_eskf/gps_vel_n_generated.rs");
    ScalarObservation { h: H, k: K, s: S }
}

pub fn gps_vel_e_observation(
    p: &[[f32; ERROR_STATES]; ERROR_STATES],
    r_vel_e: f32,
) -> ScalarObservation {
    let P = p;
    let R_VEL_E = r_vel_e;
    let mut H = [0.0; ERROR_STATES];
    let mut K = [0.0; ERROR_STATES];
    let mut S = 0.0;
    include!("generated_eskf/gps_vel_e_generated.rs");
    ScalarObservation { h: H, k: K, s: S }
}

pub fn gps_vel_d_observation(
    p: &[[f32; ERROR_STATES]; ERROR_STATES],
    r_vel_d: f32,
) -> ScalarObservation {
    let P = p;
    let R_VEL_D = r_vel_d;
    let mut H = [0.0; ERROR_STATES];
    let mut K = [0.0; ERROR_STATES];
    let mut S = 0.0;
    include!("generated_eskf/gps_vel_d_generated.rs");
    ScalarObservation { h: H, k: K, s: S }
}

pub fn stationary_accel_x_observation(
    nominal: &EskfNominalState,
    p: &[[f32; ERROR_STATES]; ERROR_STATES],
    r_stationary_accel: f32,
) -> ScalarObservation {
    let P = p;
    let R_STATIONARY_ACCEL = r_stationary_accel;
    let q0 = nominal.q0;
    let q1 = nominal.q1;
    let q2 = nominal.q2;
    let q3 = nominal.q3;
    let bax = nominal.bax;
    let g = GRAVITY_MSS;
    let mut H = [0.0; ERROR_STATES];
    let mut K = [0.0; ERROR_STATES];
    let mut S = 0.0;

    include!("generated_eskf/stationary_accel_x_generated.rs");

    ScalarObservation { h: H, k: K, s: S }
}

pub fn stationary_accel_y_observation(
    nominal: &EskfNominalState,
    p: &[[f32; ERROR_STATES]; ERROR_STATES],
    r_stationary_accel: f32,
) -> ScalarObservation {
    let P = p;
    let R_STATIONARY_ACCEL = r_stationary_accel;
    let q0 = nominal.q0;
    let q1 = nominal.q1;
    let q2 = nominal.q2;
    let q3 = nominal.q3;
    let bay = nominal.bay;
    let g = GRAVITY_MSS;
    let mut H = [0.0; ERROR_STATES];
    let mut K = [0.0; ERROR_STATES];
    let mut S = 0.0;

    include!("generated_eskf/stationary_accel_y_generated.rs");

    ScalarObservation { h: H, k: K, s: S }
}

pub fn body_vel_x_observation(
    nominal: &EskfNominalState,
    p: &[[f32; ERROR_STATES]; ERROR_STATES],
    r_body_vel: f32,
) -> ScalarObservation {
    let P = p;
    let R_BODY_VEL = r_body_vel;
    let q0 = nominal.q0;
    let q1 = nominal.q1;
    let q2 = nominal.q2;
    let q3 = nominal.q3;
    let qcs0 = nominal.qcs0;
    let qcs1 = nominal.qcs1;
    let qcs2 = nominal.qcs2;
    let qcs3 = nominal.qcs3;
    let vn = nominal.vn;
    let ve = nominal.ve;
    let vd = nominal.vd;
    let mut H = [0.0; ERROR_STATES];
    let mut K = [0.0; ERROR_STATES];
    let mut S = 0.0;

    include!("generated_eskf/body_vel_x_generated.rs");

    ScalarObservation { h: H, k: K, s: S }
}

pub fn body_vel_y_observation(
    nominal: &EskfNominalState,
    p: &[[f32; ERROR_STATES]; ERROR_STATES],
    r_body_vel: f32,
) -> ScalarObservation {
    let P = p;
    let R_BODY_VEL = r_body_vel;
    let q0 = nominal.q0;
    let q1 = nominal.q1;
    let q2 = nominal.q2;
    let q3 = nominal.q3;
    let qcs0 = nominal.qcs0;
    let qcs1 = nominal.qcs1;
    let qcs2 = nominal.qcs2;
    let qcs3 = nominal.qcs3;
    let vn = nominal.vn;
    let ve = nominal.ve;
    let vd = nominal.vd;
    let mut H = [0.0; ERROR_STATES];
    let mut K = [0.0; ERROR_STATES];
    let mut S = 0.0;

    include!("generated_eskf/body_vel_y_generated.rs");

    ScalarObservation { h: H, k: K, s: S }
}

pub fn body_vel_z_observation(
    nominal: &EskfNominalState,
    p: &[[f32; ERROR_STATES]; ERROR_STATES],
    r_body_vel: f32,
) -> ScalarObservation {
    let P = p;
    let R_BODY_VEL = r_body_vel;
    let q0 = nominal.q0;
    let q1 = nominal.q1;
    let q2 = nominal.q2;
    let q3 = nominal.q3;
    let qcs0 = nominal.qcs0;
    let qcs1 = nominal.qcs1;
    let qcs2 = nominal.qcs2;
    let qcs3 = nominal.qcs3;
    let vn = nominal.vn;
    let ve = nominal.ve;
    let vd = nominal.vd;
    let mut H = [0.0; ERROR_STATES];
    let mut K = [0.0; ERROR_STATES];
    let mut S = 0.0;

    include!("generated_eskf/body_vel_z_generated.rs");

    ScalarObservation { h: H, k: K, s: S }
}
