// Sub Expressions
const float tmp_hk_body_x0 = q0*q2;
const float tmp_hk_body_x1 = 2.0F*vn;
const float tmp_hk_body_x2 = q0*q1;
const float tmp_hk_body_x3 = q2*q3;
const float tmp_hk_body_x4 = q3*q3;
const float tmp_hk_body_x5 = q2*q2;
const float tmp_hk_body_x6 = q0*q0 - q1*q1;
const float tmp_hk_body_x7 = tmp_hk_body_x1*(q1*q3 + tmp_hk_body_x0) + 1.0F*vd*(tmp_hk_body_x4 - tmp_hk_body_x5 + tmp_hk_body_x6) - 2.0F*ve*(tmp_hk_body_x2 - tmp_hk_body_x3);
const float tmp_hk_body_x8 = q0*q3;
const float tmp_hk_body_x9 = q1*q2;
const float tmp_hk_body_x10 = -tmp_hk_body_x1*(tmp_hk_body_x8 - tmp_hk_body_x9) + 2.0F*vd*(tmp_hk_body_x2 + tmp_hk_body_x3) + 1.0F*ve*(-tmp_hk_body_x4 + tmp_hk_body_x5 + tmp_hk_body_x6);
const float tmp_hk_body_x11 = 2*tmp_hk_body_x4 + 2*tmp_hk_body_x5 - 1;
const float tmp_hk_body_x12 = tmp_hk_body_x8 + tmp_hk_body_x9;
const float tmp_hk_body_x13 = -q1*q3 + tmp_hk_body_x0;
const float tmp_hk_body_x14 = 2*tmp_hk_body_x13;
const float tmp_hk_body_x15 = -P[1][4]*tmp_hk_body_x7 + P[2][4]*tmp_hk_body_x10 - P[3][4]*tmp_hk_body_x11 + 2*P[4][4]*tmp_hk_body_x12 - P[4][5]*tmp_hk_body_x14;
const float tmp_hk_body_x16 = -P[1][5]*tmp_hk_body_x7 + P[2][5]*tmp_hk_body_x10 - P[3][5]*tmp_hk_body_x11 + 2*P[4][5]*tmp_hk_body_x12 - P[5][5]*tmp_hk_body_x14;
const float tmp_hk_body_x17 = -P[1][3]*tmp_hk_body_x7 + P[2][3]*tmp_hk_body_x10 - P[3][3]*tmp_hk_body_x11 + 2*P[3][4]*tmp_hk_body_x12 - P[3][5]*tmp_hk_body_x14;
const float tmp_hk_body_x18 = -P[1][2]*tmp_hk_body_x7 + P[2][2]*tmp_hk_body_x10 - P[2][3]*tmp_hk_body_x11 + 2*P[2][4]*tmp_hk_body_x12 - P[2][5]*tmp_hk_body_x14;
const float tmp_hk_body_x19 = -P[1][1]*tmp_hk_body_x7 + P[1][2]*tmp_hk_body_x10 - P[1][3]*tmp_hk_body_x11 + 2*P[1][4]*tmp_hk_body_x12 - P[1][5]*tmp_hk_body_x14;
const float tmp_hk_body_x20 = R_BODY_VEL + tmp_hk_body_x10*tmp_hk_body_x18 - tmp_hk_body_x11*tmp_hk_body_x17 + 2*tmp_hk_body_x12*tmp_hk_body_x15 - tmp_hk_body_x14*tmp_hk_body_x16 - tmp_hk_body_x19*tmp_hk_body_x7;
const float tmp_hk_body_x21 = 1.0F/tmp_hk_body_x20;


// Observation Jacobians
H[1] = -tmp_hk_body_x7;
H[2] = tmp_hk_body_x10;
H[3] = -tmp_hk_body_x11;
H[4] = 2*tmp_hk_body_x12;
H[5] = -2*tmp_hk_body_x13;


// Kalman gains
K[0] = tmp_hk_body_x21*(-P[0][1]*tmp_hk_body_x7 + P[0][2]*tmp_hk_body_x10 - P[0][3]*tmp_hk_body_x11 + 2*P[0][4]*tmp_hk_body_x12 - P[0][5]*tmp_hk_body_x14);
K[1] = tmp_hk_body_x19*tmp_hk_body_x21;
K[2] = tmp_hk_body_x18*tmp_hk_body_x21;
K[3] = tmp_hk_body_x17*tmp_hk_body_x21;
K[4] = tmp_hk_body_x15*tmp_hk_body_x21;
K[5] = tmp_hk_body_x16*tmp_hk_body_x21;
K[6] = tmp_hk_body_x21*(-P[1][6]*tmp_hk_body_x7 + P[2][6]*tmp_hk_body_x10 - P[3][6]*tmp_hk_body_x11 + 2*P[4][6]*tmp_hk_body_x12 - P[5][6]*tmp_hk_body_x14);
K[7] = tmp_hk_body_x21*(-P[1][7]*tmp_hk_body_x7 + P[2][7]*tmp_hk_body_x10 - P[3][7]*tmp_hk_body_x11 + 2*P[4][7]*tmp_hk_body_x12 - P[5][7]*tmp_hk_body_x14);
K[8] = tmp_hk_body_x21*(-P[1][8]*tmp_hk_body_x7 + P[2][8]*tmp_hk_body_x10 - P[3][8]*tmp_hk_body_x11 + 2*P[4][8]*tmp_hk_body_x12 - P[5][8]*tmp_hk_body_x14);
K[9] = tmp_hk_body_x21*(-P[1][9]*tmp_hk_body_x7 + P[2][9]*tmp_hk_body_x10 - P[3][9]*tmp_hk_body_x11 + 2*P[4][9]*tmp_hk_body_x12 - P[5][9]*tmp_hk_body_x14);
K[10] = tmp_hk_body_x21*(-P[1][10]*tmp_hk_body_x7 + P[2][10]*tmp_hk_body_x10 - P[3][10]*tmp_hk_body_x11 + 2*P[4][10]*tmp_hk_body_x12 - P[5][10]*tmp_hk_body_x14);
K[11] = tmp_hk_body_x21*(-P[1][11]*tmp_hk_body_x7 + P[2][11]*tmp_hk_body_x10 - P[3][11]*tmp_hk_body_x11 + 2*P[4][11]*tmp_hk_body_x12 - P[5][11]*tmp_hk_body_x14);
K[12] = tmp_hk_body_x21*(-P[1][12]*tmp_hk_body_x7 + P[2][12]*tmp_hk_body_x10 - P[3][12]*tmp_hk_body_x11 + 2*P[4][12]*tmp_hk_body_x12 - P[5][12]*tmp_hk_body_x14);
K[13] = tmp_hk_body_x21*(-P[1][13]*tmp_hk_body_x7 + P[2][13]*tmp_hk_body_x10 - P[3][13]*tmp_hk_body_x11 + 2*P[4][13]*tmp_hk_body_x12 - P[5][13]*tmp_hk_body_x14);
K[14] = tmp_hk_body_x21*(-P[1][14]*tmp_hk_body_x7 + P[2][14]*tmp_hk_body_x10 - P[3][14]*tmp_hk_body_x11 + 2*P[4][14]*tmp_hk_body_x12 - P[5][14]*tmp_hk_body_x14);
K[15] = tmp_hk_body_x21*(-P[1][15]*tmp_hk_body_x7 + P[2][15]*tmp_hk_body_x10 - P[3][15]*tmp_hk_body_x11 + 2*P[4][15]*tmp_hk_body_x12 - P[5][15]*tmp_hk_body_x14);
K[16] = tmp_hk_body_x21*(-P[1][16]*tmp_hk_body_x7 + P[2][16]*tmp_hk_body_x10 - P[3][16]*tmp_hk_body_x11 + 2*P[4][16]*tmp_hk_body_x12 - P[5][16]*tmp_hk_body_x14);
K[17] = tmp_hk_body_x21*(-P[1][17]*tmp_hk_body_x7 + P[2][17]*tmp_hk_body_x10 - P[3][17]*tmp_hk_body_x11 + 2*P[4][17]*tmp_hk_body_x12 - P[5][17]*tmp_hk_body_x14);


// Innovation Variance
S = tmp_hk_body_x20;
