// Sub Expressions
const float tmp_nhc_z_0 = q0*q1;
const float tmp_nhc_z_1 = q2*q3;
const float tmp_nhc_z_2 = tmp_nhc_z_0 + tmp_nhc_z_1;
const float tmp_nhc_z_3 = qcs0*qcs1;
const float tmp_nhc_z_4 = qcs2*qcs3;
const float tmp_nhc_z_5 = tmp_nhc_z_3 + tmp_nhc_z_4;
const float tmp_nhc_z_6 = q0*q2;
const float tmp_nhc_z_7 = q1*q3;
const float tmp_nhc_z_8 = tmp_nhc_z_6 - tmp_nhc_z_7;
const float tmp_nhc_z_9 = qcs0*qcs2;
const float tmp_nhc_z_10 = qcs1*qcs3;
const float tmp_nhc_z_11 = -tmp_nhc_z_10 + tmp_nhc_z_9;
const float tmp_nhc_z_12 = 2*q2*q2;
const float tmp_nhc_z_13 = 2*q1*q1 - 1;
const float tmp_nhc_z_14 = tmp_nhc_z_12 + tmp_nhc_z_13;
const float tmp_nhc_z_15 = 2*qcs2*qcs2;
const float tmp_nhc_z_16 = 2*qcs1*qcs1 - 1;
const float tmp_nhc_z_17 = tmp_nhc_z_15 + tmp_nhc_z_16;
const float tmp_nhc_z_18 = 4*tmp_nhc_z_11*tmp_nhc_z_8 + tmp_nhc_z_14*tmp_nhc_z_17 + 4*tmp_nhc_z_2*tmp_nhc_z_5;
const float tmp_nhc_z_19 = q0*q3;
const float tmp_nhc_z_20 = q1*q2;
const float tmp_nhc_z_21 = tmp_nhc_z_19 + tmp_nhc_z_20;
const float tmp_nhc_z_22 = 2*q3*q3;
const float tmp_nhc_z_23 = tmp_nhc_z_13 + tmp_nhc_z_22;
const float tmp_nhc_z_24 = tmp_nhc_z_0 - tmp_nhc_z_1;
const float tmp_nhc_z_25 = 2*tmp_nhc_z_11*tmp_nhc_z_21 - tmp_nhc_z_17*tmp_nhc_z_24 + tmp_nhc_z_23*tmp_nhc_z_5;
const float tmp_nhc_z_26 = 2*tmp_nhc_z_25;
const float tmp_nhc_z_27 = tmp_nhc_z_19 - tmp_nhc_z_20;
const float tmp_nhc_z_28 = tmp_nhc_z_6 + tmp_nhc_z_7;
const float tmp_nhc_z_29 = tmp_nhc_z_12 + tmp_nhc_z_22 - 1;
const float tmp_nhc_z_30 = -tmp_nhc_z_11*tmp_nhc_z_29 + tmp_nhc_z_17*tmp_nhc_z_28 + 2*tmp_nhc_z_27*tmp_nhc_z_5;
const float tmp_nhc_z_31 = 2*vn;
const float tmp_nhc_z_32 = -tmp_nhc_z_30;
const float tmp_nhc_z_33 = 2*vd;
const float tmp_nhc_z_34 = qcs0*qcs3;
const float tmp_nhc_z_35 = qcs1*qcs2;
const float tmp_nhc_z_36 = tmp_nhc_z_34 + tmp_nhc_z_35;
const float tmp_nhc_z_37 = 2*qcs3*qcs3;
const float tmp_nhc_z_38 = tmp_nhc_z_16 + tmp_nhc_z_37;
const float tmp_nhc_z_39 = tmp_nhc_z_3 - tmp_nhc_z_4;
const float tmp_nhc_z_40 = tmp_nhc_z_10 + tmp_nhc_z_9;
const float tmp_nhc_z_41 = tmp_nhc_z_34 - tmp_nhc_z_35;
const float tmp_nhc_z_42 = tmp_nhc_z_15 + tmp_nhc_z_37 - 1;


// Estimated Measurement
vc_est = tmp_nhc_z_18*vd - tmp_nhc_z_26*ve - tmp_nhc_z_30*tmp_nhc_z_31;

// Observation Jacobian
H[0] = 0;
H[1] = 0;
H[2] = 0;
H[3] = 2*tmp_nhc_z_32;
H[4] = -2*tmp_nhc_z_25;
H[5] = tmp_nhc_z_18;
H[6] = -tmp_nhc_z_18*ve - tmp_nhc_z_26*vd;
H[7] = tmp_nhc_z_18*vn - tmp_nhc_z_32*tmp_nhc_z_33;
H[8] = 2*tmp_nhc_z_25*vn - 2*tmp_nhc_z_30*ve;
H[9] = 0;
H[10] = 0;
H[11] = 0;
H[12] = 0;
H[13] = 0;
H[14] = 0;
H[15] = 0;
H[16] = 0;
H[17] = 0;
H[18] = 0;
H[19] = 0;
H[20] = 0;
H[21] = -tmp_nhc_z_31*(-tmp_nhc_z_27*tmp_nhc_z_38 + 2*tmp_nhc_z_28*tmp_nhc_z_39 + tmp_nhc_z_29*tmp_nhc_z_36) - tmp_nhc_z_33*(-tmp_nhc_z_14*tmp_nhc_z_39 + tmp_nhc_z_2*tmp_nhc_z_38 + 2*tmp_nhc_z_36*tmp_nhc_z_8) + ve*(4*tmp_nhc_z_21*tmp_nhc_z_36 + tmp_nhc_z_23*tmp_nhc_z_38 + 4*tmp_nhc_z_24*tmp_nhc_z_39);
H[22] = -tmp_nhc_z_33*(-tmp_nhc_z_14*tmp_nhc_z_40 - 2*tmp_nhc_z_2*tmp_nhc_z_41 + tmp_nhc_z_42*tmp_nhc_z_8) - 2*ve*(-tmp_nhc_z_21*tmp_nhc_z_42 + tmp_nhc_z_23*tmp_nhc_z_41 - 2*tmp_nhc_z_24*tmp_nhc_z_40) - vn*(4*tmp_nhc_z_27*tmp_nhc_z_41 + 4*tmp_nhc_z_28*tmp_nhc_z_40 + tmp_nhc_z_29*tmp_nhc_z_42);
H[23] = 0;


