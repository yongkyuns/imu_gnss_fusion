{
// Sub Expressions
let tmp_hk_vel_e0: f32 = P[4][4] + R_VEL_E;
let tmp_hk_vel_e1: f32 = 1.0_f32/tmp_hk_vel_e0;


// Observation Jacobians
H[4] = 1.0_f32;


// Kalman gains
K[0] = P[0][4]*tmp_hk_vel_e1;
K[1] = P[1][4]*tmp_hk_vel_e1;
K[2] = P[2][4]*tmp_hk_vel_e1;
K[3] = P[3][4]*tmp_hk_vel_e1;
K[4] = P[4][4]*tmp_hk_vel_e1;
K[5] = P[4][5]*tmp_hk_vel_e1;
K[6] = P[4][6]*tmp_hk_vel_e1;
K[7] = P[4][7]*tmp_hk_vel_e1;
K[8] = P[4][8]*tmp_hk_vel_e1;
K[9] = P[4][9]*tmp_hk_vel_e1;
K[10] = P[4][10]*tmp_hk_vel_e1;
K[11] = P[4][11]*tmp_hk_vel_e1;
K[12] = P[4][12]*tmp_hk_vel_e1;
K[13] = P[4][13]*tmp_hk_vel_e1;
K[14] = P[4][14]*tmp_hk_vel_e1;
K[15] = P[4][15]*tmp_hk_vel_e1;
K[16] = P[4][16]*tmp_hk_vel_e1;
K[17] = P[4][17]*tmp_hk_vel_e1;


// Innovation Variance
S = tmp_hk_vel_e0;
}
