{
// Sub Expressions
let tmp_nhc_z_0: f32 = q0*q1;
let tmp_nhc_z_1: f32 = q2*q3;
let tmp_nhc_z_2: f32 = tmp_nhc_z_0 + tmp_nhc_z_1;
let tmp_nhc_z_3: f32 = qcs0*qcs1;
let tmp_nhc_z_4: f32 = qcs2*qcs3;
let tmp_nhc_z_5: f32 = tmp_nhc_z_3 + tmp_nhc_z_4;
let tmp_nhc_z_6: f32 = q0*q2;
let tmp_nhc_z_7: f32 = q1*q3;
let tmp_nhc_z_8: f32 = tmp_nhc_z_6 - tmp_nhc_z_7;
let tmp_nhc_z_9: f32 = qcs0*qcs2;
let tmp_nhc_z_10: f32 = qcs1*qcs3;
let tmp_nhc_z_11: f32 = -tmp_nhc_z_10 + tmp_nhc_z_9;
let tmp_nhc_z_12: f32 = 2.0_f32*q2*q2;
let tmp_nhc_z_13: f32 = 2.0_f32*q1*q1 - 1.0_f32;
let tmp_nhc_z_14: f32 = tmp_nhc_z_12 + tmp_nhc_z_13;
let tmp_nhc_z_15: f32 = 2.0_f32*qcs2*qcs2;
let tmp_nhc_z_16: f32 = 2.0_f32*qcs1*qcs1 - 1.0_f32;
let tmp_nhc_z_17: f32 = tmp_nhc_z_15 + tmp_nhc_z_16;
let tmp_nhc_z_18: f32 = 4.0_f32*tmp_nhc_z_11*tmp_nhc_z_8 + tmp_nhc_z_14*tmp_nhc_z_17 + 4.0_f32*tmp_nhc_z_2*tmp_nhc_z_5;
let tmp_nhc_z_19: f32 = q0*q3;
let tmp_nhc_z_20: f32 = q1*q2;
let tmp_nhc_z_21: f32 = tmp_nhc_z_19 + tmp_nhc_z_20;
let tmp_nhc_z_22: f32 = 2.0_f32*q3*q3;
let tmp_nhc_z_23: f32 = tmp_nhc_z_13 + tmp_nhc_z_22;
let tmp_nhc_z_24: f32 = tmp_nhc_z_0 - tmp_nhc_z_1;
let tmp_nhc_z_25: f32 = 2.0_f32*tmp_nhc_z_11*tmp_nhc_z_21 - tmp_nhc_z_17*tmp_nhc_z_24 + tmp_nhc_z_23*tmp_nhc_z_5;
let tmp_nhc_z_26: f32 = 2.0_f32*tmp_nhc_z_25;
let tmp_nhc_z_27: f32 = tmp_nhc_z_19 - tmp_nhc_z_20;
let tmp_nhc_z_28: f32 = tmp_nhc_z_6 + tmp_nhc_z_7;
let tmp_nhc_z_29: f32 = tmp_nhc_z_12 + tmp_nhc_z_22 - 1.0_f32;
let tmp_nhc_z_30: f32 = -tmp_nhc_z_11*tmp_nhc_z_29 + tmp_nhc_z_17*tmp_nhc_z_28 + 2.0_f32*tmp_nhc_z_27*tmp_nhc_z_5;
let tmp_nhc_z_31: f32 = 2.0_f32*vn;
let tmp_nhc_z_32: f32 = -tmp_nhc_z_30;
let tmp_nhc_z_33: f32 = 2.0_f32*vd;
let tmp_nhc_z_34: f32 = qcs0*qcs3;
let tmp_nhc_z_35: f32 = qcs1*qcs2;
let tmp_nhc_z_36: f32 = tmp_nhc_z_34 + tmp_nhc_z_35;
let tmp_nhc_z_37: f32 = 2.0_f32*qcs3*qcs3;
let tmp_nhc_z_38: f32 = tmp_nhc_z_16 + tmp_nhc_z_37;
let tmp_nhc_z_39: f32 = tmp_nhc_z_3 - tmp_nhc_z_4;
let tmp_nhc_z_40: f32 = tmp_nhc_z_10 + tmp_nhc_z_9;
let tmp_nhc_z_41: f32 = tmp_nhc_z_34 - tmp_nhc_z_35;
let tmp_nhc_z_42: f32 = tmp_nhc_z_15 + tmp_nhc_z_37 - 1.0_f32;


// Estimated Measurement
vc_est = tmp_nhc_z_18*vd - tmp_nhc_z_26*ve - tmp_nhc_z_30*tmp_nhc_z_31;

// Observation Jacobian
H[3] = 2.0_f32*tmp_nhc_z_32;
H[4] = -2.0_f32*tmp_nhc_z_25;
H[5] = tmp_nhc_z_18;
H[6] = -tmp_nhc_z_18*ve - tmp_nhc_z_26*vd;
H[7] = tmp_nhc_z_18*vn - tmp_nhc_z_32*tmp_nhc_z_33;
H[8] = 2.0_f32*tmp_nhc_z_25*vn - 2.0_f32*tmp_nhc_z_30*ve;
H[21] = -tmp_nhc_z_31*(-tmp_nhc_z_27*tmp_nhc_z_38 + 2.0_f32*tmp_nhc_z_28*tmp_nhc_z_39 + tmp_nhc_z_29*tmp_nhc_z_36) - tmp_nhc_z_33*(-tmp_nhc_z_14*tmp_nhc_z_39 + tmp_nhc_z_2*tmp_nhc_z_38 + 2.0_f32*tmp_nhc_z_36*tmp_nhc_z_8) + ve*(4.0_f32*tmp_nhc_z_21*tmp_nhc_z_36 + tmp_nhc_z_23*tmp_nhc_z_38 + 4.0_f32*tmp_nhc_z_24*tmp_nhc_z_39);
H[22] = -tmp_nhc_z_33*(-tmp_nhc_z_14*tmp_nhc_z_40 - 2.0_f32*tmp_nhc_z_2*tmp_nhc_z_41 + tmp_nhc_z_42*tmp_nhc_z_8) - 2.0_f32*ve*(-tmp_nhc_z_21*tmp_nhc_z_42 + tmp_nhc_z_23*tmp_nhc_z_41 - 2.0_f32*tmp_nhc_z_24*tmp_nhc_z_40) - vn*(4.0_f32*tmp_nhc_z_27*tmp_nhc_z_41 + 4.0_f32*tmp_nhc_z_28*tmp_nhc_z_40 + tmp_nhc_z_29*tmp_nhc_z_42);


}
