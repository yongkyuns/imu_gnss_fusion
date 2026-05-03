{
// Sub Expressions
let tmp_nhc_y_0: f32 = q0*q1 + q2*q3;
let tmp_nhc_y_1: f32 = q0*q3 - q1*q2;
let tmp_nhc_y_2: f32 = 2.0_f32*tmp_nhc_y_1;
let tmp_nhc_y_3: f32 = 2.0_f32*q1*q1 + 2.0_f32*q3*q3 - 1.0_f32;


// Estimated Measurement
vc_est = 2.0_f32*tmp_nhc_y_0*vd - tmp_nhc_y_2*vn - tmp_nhc_y_3*ve;

// Observation Jacobian
H[3] = -2.0_f32*tmp_nhc_y_1;
H[4] = -tmp_nhc_y_3;
H[5] = 2.0_f32*tmp_nhc_y_0;
H[6] = -2.0_f32*tmp_nhc_y_0*ve - tmp_nhc_y_3*vd;
H[7] = 2.0_f32*tmp_nhc_y_0*vn + 2.0_f32*tmp_nhc_y_1*vd;
H[8] = -tmp_nhc_y_2*ve + tmp_nhc_y_3*vn;


}
