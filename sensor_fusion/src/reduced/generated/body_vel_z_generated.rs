{
// Sub Expressions
let tmp_hk_body_z0: f32 = q0*q1;
let tmp_hk_body_z1: f32 = 2.0_f32*vd;
let tmp_hk_body_z2: f32 = q0*q3;
let tmp_hk_body_z3: f32 = q1*q2;
let tmp_hk_body_z4: f32 = q2*q2;
let tmp_hk_body_z5: f32 = q1*q1;
let tmp_hk_body_z6: f32 = q0*q0 - q3*q3;
let tmp_hk_body_z7: f32 = tmp_hk_body_z1*(q2*q3 + tmp_hk_body_z0) + 1.0_f32*ve*(tmp_hk_body_z4 - tmp_hk_body_z5 + tmp_hk_body_z6) - 2.0_f32*vn*(tmp_hk_body_z2 - tmp_hk_body_z3);
let tmp_hk_body_z8: f32 = q0*q2;
let tmp_hk_body_z9: f32 = q1*q3;
let tmp_hk_body_z10: f32 = -tmp_hk_body_z1*(tmp_hk_body_z8 - tmp_hk_body_z9) + 2.0_f32*ve*(tmp_hk_body_z2 + tmp_hk_body_z3) + 1.0_f32*vn*(-tmp_hk_body_z4 + tmp_hk_body_z5 + tmp_hk_body_z6);
let tmp_hk_body_z11: f32 = tmp_hk_body_z8 + tmp_hk_body_z9;
let tmp_hk_body_z12: f32 = -q2*q3 + tmp_hk_body_z0;
let tmp_hk_body_z13: f32 = 2.0_f32*tmp_hk_body_z4 + 2.0_f32*tmp_hk_body_z5 - 1.0_f32;
let tmp_hk_body_z14: f32 = 2.0_f32*tmp_hk_body_z12;
let tmp_hk_body_z15: f32 = -P[0][0]*tmp_hk_body_z7 + P[0][1]*tmp_hk_body_z10 + 2.0_f32*P[0][3]*tmp_hk_body_z11 - P[0][4]*tmp_hk_body_z14 - P[0][5]*tmp_hk_body_z13;
let tmp_hk_body_z16: f32 = -P[0][3]*tmp_hk_body_z7 + P[1][3]*tmp_hk_body_z10 + 2.0_f32*P[3][3]*tmp_hk_body_z11 - P[3][4]*tmp_hk_body_z14 - P[3][5]*tmp_hk_body_z13;
let tmp_hk_body_z17: f32 = -P[0][4]*tmp_hk_body_z7 + P[1][4]*tmp_hk_body_z10 + 2.0_f32*P[3][4]*tmp_hk_body_z11 - P[4][4]*tmp_hk_body_z14 - P[4][5]*tmp_hk_body_z13;
let tmp_hk_body_z18: f32 = -P[0][5]*tmp_hk_body_z7 + P[1][5]*tmp_hk_body_z10 + 2.0_f32*P[3][5]*tmp_hk_body_z11 - P[4][5]*tmp_hk_body_z14 - P[5][5]*tmp_hk_body_z13;
let tmp_hk_body_z19: f32 = -P[0][1]*tmp_hk_body_z7 + P[1][1]*tmp_hk_body_z10 + 2.0_f32*P[1][3]*tmp_hk_body_z11 - P[1][4]*tmp_hk_body_z14 - P[1][5]*tmp_hk_body_z13;
let tmp_hk_body_z20: f32 = R_BODY_VEL + tmp_hk_body_z10*tmp_hk_body_z19 + 2.0_f32*tmp_hk_body_z11*tmp_hk_body_z16 - tmp_hk_body_z13*tmp_hk_body_z18 - tmp_hk_body_z14*tmp_hk_body_z17 - tmp_hk_body_z15*tmp_hk_body_z7;
let tmp_hk_body_z21: f32 = 1.0_f32/tmp_hk_body_z20;


// Observation Jacobians
H[0] = -tmp_hk_body_z7;
H[1] = tmp_hk_body_z10;
H[3] = 2.0_f32*tmp_hk_body_z11;
H[4] = -2.0_f32*tmp_hk_body_z12;
H[5] = -tmp_hk_body_z13;


// Kalman gains
K[0] = tmp_hk_body_z15*tmp_hk_body_z21;
K[1] = tmp_hk_body_z19*tmp_hk_body_z21;
K[2] = tmp_hk_body_z21*(-P[0][2]*tmp_hk_body_z7 + P[1][2]*tmp_hk_body_z10 + 2.0_f32*P[2][3]*tmp_hk_body_z11 - P[2][4]*tmp_hk_body_z14 - P[2][5]*tmp_hk_body_z13);
K[3] = tmp_hk_body_z16*tmp_hk_body_z21;
K[4] = tmp_hk_body_z17*tmp_hk_body_z21;
K[5] = tmp_hk_body_z18*tmp_hk_body_z21;
K[6] = tmp_hk_body_z21*(-P[0][6]*tmp_hk_body_z7 + P[1][6]*tmp_hk_body_z10 + 2.0_f32*P[3][6]*tmp_hk_body_z11 - P[4][6]*tmp_hk_body_z14 - P[5][6]*tmp_hk_body_z13);
K[7] = tmp_hk_body_z21*(-P[0][7]*tmp_hk_body_z7 + P[1][7]*tmp_hk_body_z10 + 2.0_f32*P[3][7]*tmp_hk_body_z11 - P[4][7]*tmp_hk_body_z14 - P[5][7]*tmp_hk_body_z13);
K[8] = tmp_hk_body_z21*(-P[0][8]*tmp_hk_body_z7 + P[1][8]*tmp_hk_body_z10 + 2.0_f32*P[3][8]*tmp_hk_body_z11 - P[4][8]*tmp_hk_body_z14 - P[5][8]*tmp_hk_body_z13);
K[9] = tmp_hk_body_z21*(-P[0][9]*tmp_hk_body_z7 + P[1][9]*tmp_hk_body_z10 + 2.0_f32*P[3][9]*tmp_hk_body_z11 - P[4][9]*tmp_hk_body_z14 - P[5][9]*tmp_hk_body_z13);
K[10] = tmp_hk_body_z21*(-P[0][10]*tmp_hk_body_z7 + P[1][10]*tmp_hk_body_z10 + 2.0_f32*P[3][10]*tmp_hk_body_z11 - P[4][10]*tmp_hk_body_z14 - P[5][10]*tmp_hk_body_z13);
K[11] = tmp_hk_body_z21*(-P[0][11]*tmp_hk_body_z7 + P[1][11]*tmp_hk_body_z10 + 2.0_f32*P[3][11]*tmp_hk_body_z11 - P[4][11]*tmp_hk_body_z14 - P[5][11]*tmp_hk_body_z13);
K[12] = tmp_hk_body_z21*(-P[0][12]*tmp_hk_body_z7 + P[1][12]*tmp_hk_body_z10 + 2.0_f32*P[3][12]*tmp_hk_body_z11 - P[4][12]*tmp_hk_body_z14 - P[5][12]*tmp_hk_body_z13);
K[13] = tmp_hk_body_z21*(-P[0][13]*tmp_hk_body_z7 + P[1][13]*tmp_hk_body_z10 + 2.0_f32*P[3][13]*tmp_hk_body_z11 - P[4][13]*tmp_hk_body_z14 - P[5][13]*tmp_hk_body_z13);
K[14] = tmp_hk_body_z21*(-P[0][14]*tmp_hk_body_z7 + P[1][14]*tmp_hk_body_z10 + 2.0_f32*P[3][14]*tmp_hk_body_z11 - P[4][14]*tmp_hk_body_z14 - P[5][14]*tmp_hk_body_z13);
K[15] = tmp_hk_body_z21*(-P[0][15]*tmp_hk_body_z7 + P[1][15]*tmp_hk_body_z10 + 2.0_f32*P[3][15]*tmp_hk_body_z11 - P[4][15]*tmp_hk_body_z14 - P[5][15]*tmp_hk_body_z13);
K[16] = tmp_hk_body_z21*(-P[0][16]*tmp_hk_body_z7 + P[1][16]*tmp_hk_body_z10 + 2.0_f32*P[3][16]*tmp_hk_body_z11 - P[4][16]*tmp_hk_body_z14 - P[5][16]*tmp_hk_body_z13);
K[17] = tmp_hk_body_z21*(-P[0][17]*tmp_hk_body_z7 + P[1][17]*tmp_hk_body_z10 + 2.0_f32*P[3][17]*tmp_hk_body_z11 - P[4][17]*tmp_hk_body_z14 - P[5][17]*tmp_hk_body_z13);


// Innovation Variance
S = tmp_hk_body_z20;
}
