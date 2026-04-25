#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused_assignments)]

use crate::loose::{LOOSE_ERROR_STATES, LOOSE_NOISE_STATES, LooseImuDelta, LooseNominalState};

pub fn error_transition(
    nominal: &LooseNominalState,
    imu: LooseImuDelta,
) -> (
    [[f32; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES],
    [[f32; LOOSE_NOISE_STATES]; LOOSE_ERROR_STATES],
) {
    let mut f = [[0.0; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES];
    let mut noise_input = [[0.0; LOOSE_NOISE_STATES]; LOOSE_ERROR_STATES];
    let F = &mut f;
    let G = &mut noise_input;

    let q0 = nominal.q0;
    let q1 = nominal.q1;
    let q2 = nominal.q2;
    let q3 = nominal.q3;
    let bax = nominal.bax;
    let bay = nominal.bay;
    let baz = nominal.baz;
    let sax = nominal.sax;
    let say = nominal.say;
    let saz = nominal.saz;
    let dax = imu.dax_2;
    let day = imu.day_2;
    let daz = imu.daz_2;
    let dvx = imu.dvx_2;
    let dvy = imu.dvy_2;
    let dvz = imu.dvz_2;
    let dt = imu.dt;

    include!("generated_loose/reference_error_noise_input_generated.rs");
    include!("generated_loose/reference_error_transition_generated.rs");

    (f, noise_input)
}

pub fn nhc_y(nominal: &LooseNominalState) -> (f32, [f32; LOOSE_ERROR_STATES]) {
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
    let mut vc_est = 0.0;
    let mut H = [0.0; LOOSE_ERROR_STATES];

    include!("generated_loose/reference_nhc_y_generated.rs");

    (vc_est, H)
}

pub fn nhc_z(nominal: &LooseNominalState) -> (f32, [f32; LOOSE_ERROR_STATES]) {
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
    let mut vc_est = 0.0;
    let mut H = [0.0; LOOSE_ERROR_STATES];

    include!("generated_loose/reference_nhc_z_generated.rs");

    (vc_est, H)
}
