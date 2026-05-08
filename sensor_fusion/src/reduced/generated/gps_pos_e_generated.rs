{
// Sub Expressions
let tmp_hk_pos_e0: f32 = P[7][7] + R_POS_E;
let tmp_hk_pos_e1: f32 = 1.0_f32/tmp_hk_pos_e0;


// Observation Jacobians
H[7] = 1.0_f32;


// Kalman gains
K[0] = P[0][7]*tmp_hk_pos_e1;
K[1] = P[1][7]*tmp_hk_pos_e1;
K[2] = P[2][7]*tmp_hk_pos_e1;
K[3] = P[3][7]*tmp_hk_pos_e1;
K[4] = P[4][7]*tmp_hk_pos_e1;
K[5] = P[5][7]*tmp_hk_pos_e1;
K[6] = P[6][7]*tmp_hk_pos_e1;
K[7] = P[7][7]*tmp_hk_pos_e1;
K[8] = P[7][8]*tmp_hk_pos_e1;
K[9] = P[7][9]*tmp_hk_pos_e1;
K[10] = P[7][10]*tmp_hk_pos_e1;
K[11] = P[7][11]*tmp_hk_pos_e1;
K[12] = P[7][12]*tmp_hk_pos_e1;
K[13] = P[7][13]*tmp_hk_pos_e1;
K[14] = P[7][14]*tmp_hk_pos_e1;
K[15] = P[7][15]*tmp_hk_pos_e1;
K[16] = P[7][16]*tmp_hk_pos_e1;
K[17] = P[7][17]*tmp_hk_pos_e1;


// Innovation Variance
S = tmp_hk_pos_e0;
}
