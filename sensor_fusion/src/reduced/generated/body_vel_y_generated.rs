{
// Sub Expressions
let tmp_hk_body_y0: f32 = q0*q2;
let tmp_hk_body_y1: f32 = q1*q3;
let tmp_hk_body_y2: f32 = q0*q1;
let tmp_hk_body_y3: f32 = q2*q3;
let tmp_hk_body_y4: f32 = 2.0_f32*ve;
let tmp_hk_body_y5: f32 = q3*q3;
let tmp_hk_body_y6: f32 = q1*q1;
let tmp_hk_body_y7: f32 = q0*q0 - q2*q2;
let tmp_hk_body_y8: f32 = -tmp_hk_body_y4*(tmp_hk_body_y2 - tmp_hk_body_y3) + 1.0_f32*vd*(tmp_hk_body_y5 - tmp_hk_body_y6 + tmp_hk_body_y7) + 2.0_f32*vn*(tmp_hk_body_y0 + tmp_hk_body_y1);
let tmp_hk_body_y9: f32 = q0*q3;
let tmp_hk_body_y10: f32 = tmp_hk_body_y4*(q1*q2 + tmp_hk_body_y9) - 2.0_f32*vd*(tmp_hk_body_y0 - tmp_hk_body_y1) + 1.0_f32*vn*(-tmp_hk_body_y5 + tmp_hk_body_y6 + tmp_hk_body_y7);
let tmp_hk_body_y11: f32 = -q1*q2 + tmp_hk_body_y9;
let tmp_hk_body_y12: f32 = 2.0_f32*tmp_hk_body_y5 + 2.0_f32*tmp_hk_body_y6 - 1.0_f32;
let tmp_hk_body_y13: f32 = tmp_hk_body_y2 + tmp_hk_body_y3;
let tmp_hk_body_y14: f32 = 2.0_f32*tmp_hk_body_y11;
let tmp_hk_body_y15: f32 = P[0][0]*tmp_hk_body_y8 - P[0][2]*tmp_hk_body_y10 - P[0][3]*tmp_hk_body_y14 - P[0][4]*tmp_hk_body_y12 + 2.0_f32*P[0][5]*tmp_hk_body_y13;
let tmp_hk_body_y16: f32 = P[0][5]*tmp_hk_body_y8 - P[2][5]*tmp_hk_body_y10 - P[3][5]*tmp_hk_body_y14 - P[4][5]*tmp_hk_body_y12 + 2.0_f32*P[5][5]*tmp_hk_body_y13;
let tmp_hk_body_y17: f32 = P[0][3]*tmp_hk_body_y8 - P[2][3]*tmp_hk_body_y10 - P[3][3]*tmp_hk_body_y14 - P[3][4]*tmp_hk_body_y12 + 2.0_f32*P[3][5]*tmp_hk_body_y13;
let tmp_hk_body_y18: f32 = P[0][4]*tmp_hk_body_y8 - P[2][4]*tmp_hk_body_y10 - P[3][4]*tmp_hk_body_y14 - P[4][4]*tmp_hk_body_y12 + 2.0_f32*P[4][5]*tmp_hk_body_y13;
let tmp_hk_body_y19: f32 = P[0][2]*tmp_hk_body_y8 - P[2][2]*tmp_hk_body_y10 - P[2][3]*tmp_hk_body_y14 - P[2][4]*tmp_hk_body_y12 + 2.0_f32*P[2][5]*tmp_hk_body_y13;
let tmp_hk_body_y20: f32 = R_BODY_VEL - tmp_hk_body_y10*tmp_hk_body_y19 - tmp_hk_body_y12*tmp_hk_body_y18 + 2.0_f32*tmp_hk_body_y13*tmp_hk_body_y16 - tmp_hk_body_y14*tmp_hk_body_y17 + tmp_hk_body_y15*tmp_hk_body_y8;
let tmp_hk_body_y21: f32 = 1.0_f32/tmp_hk_body_y20;


// Observation Jacobians
H[0] = tmp_hk_body_y8;
H[2] = -tmp_hk_body_y10;
H[3] = -2.0_f32*tmp_hk_body_y11;
H[4] = -tmp_hk_body_y12;
H[5] = 2.0_f32*tmp_hk_body_y13;


// Kalman gains
K[0] = tmp_hk_body_y15*tmp_hk_body_y21;
K[1] = tmp_hk_body_y21*(P[0][1]*tmp_hk_body_y8 - P[1][2]*tmp_hk_body_y10 - P[1][3]*tmp_hk_body_y14 - P[1][4]*tmp_hk_body_y12 + 2.0_f32*P[1][5]*tmp_hk_body_y13);
K[2] = tmp_hk_body_y19*tmp_hk_body_y21;
K[3] = tmp_hk_body_y17*tmp_hk_body_y21;
K[4] = tmp_hk_body_y18*tmp_hk_body_y21;
K[5] = tmp_hk_body_y16*tmp_hk_body_y21;
K[6] = tmp_hk_body_y21*(P[0][6]*tmp_hk_body_y8 - P[2][6]*tmp_hk_body_y10 - P[3][6]*tmp_hk_body_y14 - P[4][6]*tmp_hk_body_y12 + 2.0_f32*P[5][6]*tmp_hk_body_y13);
K[7] = tmp_hk_body_y21*(P[0][7]*tmp_hk_body_y8 - P[2][7]*tmp_hk_body_y10 - P[3][7]*tmp_hk_body_y14 - P[4][7]*tmp_hk_body_y12 + 2.0_f32*P[5][7]*tmp_hk_body_y13);
K[8] = tmp_hk_body_y21*(P[0][8]*tmp_hk_body_y8 - P[2][8]*tmp_hk_body_y10 - P[3][8]*tmp_hk_body_y14 - P[4][8]*tmp_hk_body_y12 + 2.0_f32*P[5][8]*tmp_hk_body_y13);
K[9] = tmp_hk_body_y21*(P[0][9]*tmp_hk_body_y8 - P[2][9]*tmp_hk_body_y10 - P[3][9]*tmp_hk_body_y14 - P[4][9]*tmp_hk_body_y12 + 2.0_f32*P[5][9]*tmp_hk_body_y13);
K[10] = tmp_hk_body_y21*(P[0][10]*tmp_hk_body_y8 - P[2][10]*tmp_hk_body_y10 - P[3][10]*tmp_hk_body_y14 - P[4][10]*tmp_hk_body_y12 + 2.0_f32*P[5][10]*tmp_hk_body_y13);
K[11] = tmp_hk_body_y21*(P[0][11]*tmp_hk_body_y8 - P[2][11]*tmp_hk_body_y10 - P[3][11]*tmp_hk_body_y14 - P[4][11]*tmp_hk_body_y12 + 2.0_f32*P[5][11]*tmp_hk_body_y13);
K[12] = tmp_hk_body_y21*(P[0][12]*tmp_hk_body_y8 - P[2][12]*tmp_hk_body_y10 - P[3][12]*tmp_hk_body_y14 - P[4][12]*tmp_hk_body_y12 + 2.0_f32*P[5][12]*tmp_hk_body_y13);
K[13] = tmp_hk_body_y21*(P[0][13]*tmp_hk_body_y8 - P[2][13]*tmp_hk_body_y10 - P[3][13]*tmp_hk_body_y14 - P[4][13]*tmp_hk_body_y12 + 2.0_f32*P[5][13]*tmp_hk_body_y13);
K[14] = tmp_hk_body_y21*(P[0][14]*tmp_hk_body_y8 - P[2][14]*tmp_hk_body_y10 - P[3][14]*tmp_hk_body_y14 - P[4][14]*tmp_hk_body_y12 + 2.0_f32*P[5][14]*tmp_hk_body_y13);
K[15] = tmp_hk_body_y21*(P[0][15]*tmp_hk_body_y8 - P[2][15]*tmp_hk_body_y10 - P[3][15]*tmp_hk_body_y14 - P[4][15]*tmp_hk_body_y12 + 2.0_f32*P[5][15]*tmp_hk_body_y13);
K[16] = tmp_hk_body_y21*(P[0][16]*tmp_hk_body_y8 - P[2][16]*tmp_hk_body_y10 - P[3][16]*tmp_hk_body_y14 - P[4][16]*tmp_hk_body_y12 + 2.0_f32*P[5][16]*tmp_hk_body_y13);
K[17] = tmp_hk_body_y21*(P[0][17]*tmp_hk_body_y8 - P[2][17]*tmp_hk_body_y10 - P[3][17]*tmp_hk_body_y14 - P[4][17]*tmp_hk_body_y12 + 2.0_f32*P[5][17]*tmp_hk_body_y13);


// Innovation Variance
S = tmp_hk_body_y20;
}
