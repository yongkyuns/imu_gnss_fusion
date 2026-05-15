{
// Sub Expressions
let tmp_hk_vel_n0: f32 = P[3][3] + R_VEL_N;
let tmp_hk_vel_n1: f32 = 1.0_f32/tmp_hk_vel_n0;


// Observation Jacobians
H[3] = 1.0_f32;


// Kalman gains
K[0] = P[0][3]*tmp_hk_vel_n1;
K[1] = P[1][3]*tmp_hk_vel_n1;
K[2] = P[2][3]*tmp_hk_vel_n1;
K[3] = P[3][3]*tmp_hk_vel_n1;
K[4] = P[3][4]*tmp_hk_vel_n1;
K[5] = P[3][5]*tmp_hk_vel_n1;
K[6] = P[3][6]*tmp_hk_vel_n1;
K[7] = P[3][7]*tmp_hk_vel_n1;
K[8] = P[3][8]*tmp_hk_vel_n1;
K[9] = P[3][9]*tmp_hk_vel_n1;
K[10] = P[3][10]*tmp_hk_vel_n1;
K[11] = P[3][11]*tmp_hk_vel_n1;
K[12] = P[3][12]*tmp_hk_vel_n1;
K[13] = P[3][13]*tmp_hk_vel_n1;
K[14] = P[3][14]*tmp_hk_vel_n1;
K[15] = P[3][15]*tmp_hk_vel_n1;
K[16] = P[3][16]*tmp_hk_vel_n1;
K[17] = P[3][17]*tmp_hk_vel_n1;


// Innovation Variance
S = tmp_hk_vel_n0;
}
