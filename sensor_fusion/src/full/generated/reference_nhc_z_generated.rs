{
// Sub Expressions
let tmp_nhc_z_0: f32 = q0*q2 + q1*q3;
let tmp_nhc_z_1: f32 = q0*q1 - q2*q3;
let tmp_nhc_z_2: f32 = 2.0_f32*tmp_nhc_z_1;
let tmp_nhc_z_3: f32 = 2.0_f32*q1*q1 + 2.0_f32*q2*q2 - 1.0_f32;


// Estimated Measurement
vc_est = 2.0_f32*tmp_nhc_z_0*vn - tmp_nhc_z_2*ve - tmp_nhc_z_3*vd;

// Observation Jacobian
H[3] = 2.0_f32*tmp_nhc_z_0;
H[4] = -2.0_f32*tmp_nhc_z_1;
H[5] = -tmp_nhc_z_3;
H[6] = -tmp_nhc_z_2*vd + tmp_nhc_z_3*ve;
H[7] = -2.0_f32*tmp_nhc_z_0*vd - tmp_nhc_z_3*vn;
H[8] = 2.0_f32*tmp_nhc_z_0*ve + 2.0_f32*tmp_nhc_z_1*vn;


}
