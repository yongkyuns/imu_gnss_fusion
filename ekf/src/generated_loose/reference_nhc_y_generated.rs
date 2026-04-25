{
// Sub Expressions
let tmp_nhc_y_0: f32 = qcs0*qcs3;
let tmp_nhc_y_1: f32 = qcs1*qcs2;
let tmp_nhc_y_2: f32 = tmp_nhc_y_0 + tmp_nhc_y_1;
let tmp_nhc_y_3: f32 = q0*q2;
let tmp_nhc_y_4: f32 = q1*q3;
let tmp_nhc_y_5: f32 = tmp_nhc_y_3 - tmp_nhc_y_4;
let tmp_nhc_y_6: f32 = q0*q1;
let tmp_nhc_y_7: f32 = q2*q3;
let tmp_nhc_y_8: f32 = tmp_nhc_y_6 + tmp_nhc_y_7;
let tmp_nhc_y_9: f32 = 2.0_f32*qcs3*qcs3;
let tmp_nhc_y_10: f32 = 2.0_f32*qcs1*qcs1 - 1.0_f32;
let tmp_nhc_y_11: f32 = tmp_nhc_y_10 + tmp_nhc_y_9;
let tmp_nhc_y_12: f32 = qcs0*qcs1;
let tmp_nhc_y_13: f32 = qcs2*qcs3;
let tmp_nhc_y_14: f32 = tmp_nhc_y_12 - tmp_nhc_y_13;
let tmp_nhc_y_15: f32 = 2.0_f32*q2*q2;
let tmp_nhc_y_16: f32 = 2.0_f32*q1*q1 - 1.0_f32;
let tmp_nhc_y_17: f32 = tmp_nhc_y_15 + tmp_nhc_y_16;
let tmp_nhc_y_18: f32 = tmp_nhc_y_11*tmp_nhc_y_8 - tmp_nhc_y_14*tmp_nhc_y_17 + 2.0_f32*tmp_nhc_y_2*tmp_nhc_y_5;
let tmp_nhc_y_19: f32 = 2.0_f32*tmp_nhc_y_18;
let tmp_nhc_y_20: f32 = q0*q3;
let tmp_nhc_y_21: f32 = q1*q2;
let tmp_nhc_y_22: f32 = tmp_nhc_y_20 + tmp_nhc_y_21;
let tmp_nhc_y_23: f32 = tmp_nhc_y_6 - tmp_nhc_y_7;
let tmp_nhc_y_24: f32 = 2.0_f32*q3*q3;
let tmp_nhc_y_25: f32 = tmp_nhc_y_16 + tmp_nhc_y_24;
let tmp_nhc_y_26: f32 = tmp_nhc_y_11*tmp_nhc_y_25 + 4.0_f32*tmp_nhc_y_14*tmp_nhc_y_23 + 4.0_f32*tmp_nhc_y_2*tmp_nhc_y_22;
let tmp_nhc_y_27: f32 = tmp_nhc_y_3 + tmp_nhc_y_4;
let tmp_nhc_y_28: f32 = tmp_nhc_y_15 + tmp_nhc_y_24 - 1.0_f32;
let tmp_nhc_y_29: f32 = tmp_nhc_y_20 - tmp_nhc_y_21;
let tmp_nhc_y_30: f32 = -tmp_nhc_y_11*tmp_nhc_y_29 + 2.0_f32*tmp_nhc_y_14*tmp_nhc_y_27 + tmp_nhc_y_2*tmp_nhc_y_28;
let tmp_nhc_y_31: f32 = 2.0_f32*tmp_nhc_y_30;
let tmp_nhc_y_32: f32 = -tmp_nhc_y_30;
let tmp_nhc_y_33: f32 = -tmp_nhc_y_18;
let tmp_nhc_y_34: f32 = tmp_nhc_y_12 + tmp_nhc_y_13;
let tmp_nhc_y_35: f32 = qcs0*qcs2;
let tmp_nhc_y_36: f32 = qcs1*qcs3;
let tmp_nhc_y_37: f32 = tmp_nhc_y_35 - tmp_nhc_y_36;
let tmp_nhc_y_38: f32 = 2.0_f32*qcs2*qcs2;
let tmp_nhc_y_39: f32 = tmp_nhc_y_10 + tmp_nhc_y_38;
let tmp_nhc_y_40: f32 = 2.0_f32*ve;
let tmp_nhc_y_41: f32 = tmp_nhc_y_0 - tmp_nhc_y_1;
let tmp_nhc_y_42: f32 = tmp_nhc_y_35 + tmp_nhc_y_36;
let tmp_nhc_y_43: f32 = tmp_nhc_y_38 + tmp_nhc_y_9 - 1.0_f32;


// Estimated Measurement
vc_est = -tmp_nhc_y_19*vd + tmp_nhc_y_26*ve - tmp_nhc_y_31*vn;

// Observation Jacobian
H[0] = 0.0_f32;
H[1] = 0.0_f32;
H[2] = 0.0_f32;
H[3] = 2.0_f32*tmp_nhc_y_32;
H[4] = tmp_nhc_y_26;
H[5] = 2.0_f32*tmp_nhc_y_33;
H[6] = tmp_nhc_y_19*ve + tmp_nhc_y_26*vd;
H[7] = -2.0_f32*tmp_nhc_y_32*vd + 2.0_f32*tmp_nhc_y_33*vn;
H[8] = -tmp_nhc_y_26*vn - tmp_nhc_y_31*ve;
H[9] = 0.0_f32;
H[10] = 0.0_f32;
H[11] = 0.0_f32;
H[12] = 0.0_f32;
H[13] = 0.0_f32;
H[14] = 0.0_f32;
H[15] = 0.0_f32;
H[16] = 0.0_f32;
H[17] = 0.0_f32;
H[18] = 0.0_f32;
H[19] = 0.0_f32;
H[20] = 0.0_f32;
H[21] = -tmp_nhc_y_40*(-2.0_f32*tmp_nhc_y_22*tmp_nhc_y_37 + tmp_nhc_y_23*tmp_nhc_y_39 - tmp_nhc_y_25*tmp_nhc_y_34) - vd*(tmp_nhc_y_17*tmp_nhc_y_39 + 4.0_f32*tmp_nhc_y_34*tmp_nhc_y_8 + 4.0_f32*tmp_nhc_y_37*tmp_nhc_y_5) - 2.0_f32*vn*(-tmp_nhc_y_27*tmp_nhc_y_39 + tmp_nhc_y_28*tmp_nhc_y_37 - 2.0_f32*tmp_nhc_y_29*tmp_nhc_y_34);
H[22] = 0.0_f32;
H[23] = -tmp_nhc_y_40*(tmp_nhc_y_22*tmp_nhc_y_43 + 2.0_f32*tmp_nhc_y_23*tmp_nhc_y_42 - tmp_nhc_y_25*tmp_nhc_y_41) - 2.0_f32*vd*(tmp_nhc_y_17*tmp_nhc_y_42 + 2.0_f32*tmp_nhc_y_41*tmp_nhc_y_8 - tmp_nhc_y_43*tmp_nhc_y_5) + vn*(4.0_f32*tmp_nhc_y_27*tmp_nhc_y_42 + tmp_nhc_y_28*tmp_nhc_y_43 + 4.0_f32*tmp_nhc_y_29*tmp_nhc_y_41);


}
