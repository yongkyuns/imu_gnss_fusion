{
// Sub Expressions
let tmp_hk_vel_d0: f32 = P[5][5] + R_VEL_D;
let tmp_hk_vel_d1: f32 = 1.0_f32/tmp_hk_vel_d0;


// Observation Jacobians
H[5] = 1.0_f32;


// Kalman gains
K[0] = P[0][5]*tmp_hk_vel_d1;
K[1] = P[1][5]*tmp_hk_vel_d1;
K[2] = P[2][5]*tmp_hk_vel_d1;
K[3] = P[3][5]*tmp_hk_vel_d1;
K[4] = P[4][5]*tmp_hk_vel_d1;
K[5] = P[5][5]*tmp_hk_vel_d1;
K[6] = P[5][6]*tmp_hk_vel_d1;
K[7] = P[5][7]*tmp_hk_vel_d1;
K[8] = P[5][8]*tmp_hk_vel_d1;
K[9] = P[5][9]*tmp_hk_vel_d1;
K[10] = P[5][10]*tmp_hk_vel_d1;
K[11] = P[5][11]*tmp_hk_vel_d1;
K[12] = P[5][12]*tmp_hk_vel_d1;
K[13] = P[5][13]*tmp_hk_vel_d1;
K[14] = P[5][14]*tmp_hk_vel_d1;
K[15] = P[5][15]*tmp_hk_vel_d1;
K[16] = P[5][16]*tmp_hk_vel_d1;
K[17] = P[5][17]*tmp_hk_vel_d1;


// Innovation Variance
S = tmp_hk_vel_d0;
}
