// Sub Expressions
const float tmp_nhc_y_0 = qcs0*qcs3;
const float tmp_nhc_y_1 = qcs1*qcs2;
const float tmp_nhc_y_2 = tmp_nhc_y_0 + tmp_nhc_y_1;
const float tmp_nhc_y_3 = q0*q2;
const float tmp_nhc_y_4 = q1*q3;
const float tmp_nhc_y_5 = tmp_nhc_y_3 - tmp_nhc_y_4;
const float tmp_nhc_y_6 = q0*q1;
const float tmp_nhc_y_7 = q2*q3;
const float tmp_nhc_y_8 = tmp_nhc_y_6 + tmp_nhc_y_7;
const float tmp_nhc_y_9 = 2*qcs3*qcs3;
const float tmp_nhc_y_10 = 2*qcs1*qcs1 - 1;
const float tmp_nhc_y_11 = tmp_nhc_y_10 + tmp_nhc_y_9;
const float tmp_nhc_y_12 = qcs0*qcs1;
const float tmp_nhc_y_13 = qcs2*qcs3;
const float tmp_nhc_y_14 = tmp_nhc_y_12 - tmp_nhc_y_13;
const float tmp_nhc_y_15 = 2*q2*q2;
const float tmp_nhc_y_16 = 2*q1*q1 - 1;
const float tmp_nhc_y_17 = tmp_nhc_y_15 + tmp_nhc_y_16;
const float tmp_nhc_y_18 = tmp_nhc_y_11*tmp_nhc_y_8 - tmp_nhc_y_14*tmp_nhc_y_17 + 2*tmp_nhc_y_2*tmp_nhc_y_5;
const float tmp_nhc_y_19 = 2*tmp_nhc_y_18;
const float tmp_nhc_y_20 = q0*q3;
const float tmp_nhc_y_21 = q1*q2;
const float tmp_nhc_y_22 = tmp_nhc_y_20 + tmp_nhc_y_21;
const float tmp_nhc_y_23 = tmp_nhc_y_6 - tmp_nhc_y_7;
const float tmp_nhc_y_24 = 2*q3*q3;
const float tmp_nhc_y_25 = tmp_nhc_y_16 + tmp_nhc_y_24;
const float tmp_nhc_y_26 = tmp_nhc_y_11*tmp_nhc_y_25 + 4*tmp_nhc_y_14*tmp_nhc_y_23 + 4*tmp_nhc_y_2*tmp_nhc_y_22;
const float tmp_nhc_y_27 = tmp_nhc_y_3 + tmp_nhc_y_4;
const float tmp_nhc_y_28 = tmp_nhc_y_15 + tmp_nhc_y_24 - 1;
const float tmp_nhc_y_29 = tmp_nhc_y_20 - tmp_nhc_y_21;
const float tmp_nhc_y_30 = -tmp_nhc_y_11*tmp_nhc_y_29 + 2*tmp_nhc_y_14*tmp_nhc_y_27 + tmp_nhc_y_2*tmp_nhc_y_28;
const float tmp_nhc_y_31 = 2*tmp_nhc_y_30;
const float tmp_nhc_y_32 = -tmp_nhc_y_30;
const float tmp_nhc_y_33 = -tmp_nhc_y_18;
const float tmp_nhc_y_34 = tmp_nhc_y_12 + tmp_nhc_y_13;
const float tmp_nhc_y_35 = qcs0*qcs2;
const float tmp_nhc_y_36 = qcs1*qcs3;
const float tmp_nhc_y_37 = tmp_nhc_y_35 - tmp_nhc_y_36;
const float tmp_nhc_y_38 = 2*qcs2*qcs2;
const float tmp_nhc_y_39 = tmp_nhc_y_10 + tmp_nhc_y_38;
const float tmp_nhc_y_40 = 2*ve;
const float tmp_nhc_y_41 = tmp_nhc_y_0 - tmp_nhc_y_1;
const float tmp_nhc_y_42 = tmp_nhc_y_35 + tmp_nhc_y_36;
const float tmp_nhc_y_43 = tmp_nhc_y_38 + tmp_nhc_y_9 - 1;


// Estimated Measurement
vc_est = -tmp_nhc_y_19*vd + tmp_nhc_y_26*ve - tmp_nhc_y_31*vn;

// Observation Jacobian
H[0] = 0;
H[1] = 0;
H[2] = 0;
H[3] = 2*tmp_nhc_y_32;
H[4] = tmp_nhc_y_26;
H[5] = 2*tmp_nhc_y_33;
H[6] = tmp_nhc_y_19*ve + tmp_nhc_y_26*vd;
H[7] = -2*tmp_nhc_y_32*vd + 2*tmp_nhc_y_33*vn;
H[8] = -tmp_nhc_y_26*vn - tmp_nhc_y_31*ve;
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
H[21] = -tmp_nhc_y_40*(-2*tmp_nhc_y_22*tmp_nhc_y_37 + tmp_nhc_y_23*tmp_nhc_y_39 - tmp_nhc_y_25*tmp_nhc_y_34) - vd*(tmp_nhc_y_17*tmp_nhc_y_39 + 4*tmp_nhc_y_34*tmp_nhc_y_8 + 4*tmp_nhc_y_37*tmp_nhc_y_5) - 2*vn*(-tmp_nhc_y_27*tmp_nhc_y_39 + tmp_nhc_y_28*tmp_nhc_y_37 - 2*tmp_nhc_y_29*tmp_nhc_y_34);
H[22] = 0;
H[23] = -tmp_nhc_y_40*(tmp_nhc_y_22*tmp_nhc_y_43 + 2*tmp_nhc_y_23*tmp_nhc_y_42 - tmp_nhc_y_25*tmp_nhc_y_41) - 2*vd*(tmp_nhc_y_17*tmp_nhc_y_42 + 2*tmp_nhc_y_41*tmp_nhc_y_8 - tmp_nhc_y_43*tmp_nhc_y_5) + vn*(4*tmp_nhc_y_27*tmp_nhc_y_42 + tmp_nhc_y_28*tmp_nhc_y_43 + 4*tmp_nhc_y_29*tmp_nhc_y_41);


