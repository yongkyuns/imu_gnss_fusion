{
// Sub Expressions
let tmp_hk_pos_d0: f32 = P[8][8] + R_POS_D;
let tmp_hk_pos_d1: f32 = 1.0_f32/tmp_hk_pos_d0;


// Observation Jacobians
H[8] = 1.0_f32;


// Kalman gains
K[0] = P[0][8]*tmp_hk_pos_d1;
K[1] = P[1][8]*tmp_hk_pos_d1;
K[2] = P[2][8]*tmp_hk_pos_d1;
K[3] = P[3][8]*tmp_hk_pos_d1;
K[4] = P[4][8]*tmp_hk_pos_d1;
K[5] = P[5][8]*tmp_hk_pos_d1;
K[6] = P[6][8]*tmp_hk_pos_d1;
K[7] = P[7][8]*tmp_hk_pos_d1;
K[8] = P[8][8]*tmp_hk_pos_d1;
K[9] = P[8][9]*tmp_hk_pos_d1;
K[10] = P[8][10]*tmp_hk_pos_d1;
K[11] = P[8][11]*tmp_hk_pos_d1;
K[12] = P[8][12]*tmp_hk_pos_d1;
K[13] = P[8][13]*tmp_hk_pos_d1;
K[14] = P[8][14]*tmp_hk_pos_d1;
K[15] = P[8][15]*tmp_hk_pos_d1;
K[16] = P[8][16]*tmp_hk_pos_d1;
K[17] = P[8][17]*tmp_hk_pos_d1;


// Innovation Variance
S = tmp_hk_pos_d0;
}
