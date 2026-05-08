{
// Sub Expressions
let tmp_hk_pos_n0: f32 = P[6][6] + R_POS_N;
let tmp_hk_pos_n1: f32 = 1.0_f32/tmp_hk_pos_n0;


// Observation Jacobians
H[6] = 1.0_f32;


// Kalman gains
K[0] = P[0][6]*tmp_hk_pos_n1;
K[1] = P[1][6]*tmp_hk_pos_n1;
K[2] = P[2][6]*tmp_hk_pos_n1;
K[3] = P[3][6]*tmp_hk_pos_n1;
K[4] = P[4][6]*tmp_hk_pos_n1;
K[5] = P[5][6]*tmp_hk_pos_n1;
K[6] = P[6][6]*tmp_hk_pos_n1;
K[7] = P[6][7]*tmp_hk_pos_n1;
K[8] = P[6][8]*tmp_hk_pos_n1;
K[9] = P[6][9]*tmp_hk_pos_n1;
K[10] = P[6][10]*tmp_hk_pos_n1;
K[11] = P[6][11]*tmp_hk_pos_n1;
K[12] = P[6][12]*tmp_hk_pos_n1;
K[13] = P[6][13]*tmp_hk_pos_n1;
K[14] = P[6][14]*tmp_hk_pos_n1;
K[15] = P[6][15]*tmp_hk_pos_n1;
K[16] = P[6][16]*tmp_hk_pos_n1;
K[17] = P[6][17]*tmp_hk_pos_n1;


// Innovation Variance
S = tmp_hk_pos_n0;
}
