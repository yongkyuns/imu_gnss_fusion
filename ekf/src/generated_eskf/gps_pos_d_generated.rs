{
// Sub Expressions
let ESKF_HK_POS_D0: f32 = P[8][8] + R_POS_D;
let ESKF_HK_POS_D1: f32 = 1.0_f32/ESKF_HK_POS_D0;


// Observation Jacobians
H[0] = 0.0_f32;
H[1] = 0.0_f32;
H[2] = 0.0_f32;
H[3] = 0.0_f32;
H[4] = 0.0_f32;
H[5] = 0.0_f32;
H[6] = 0.0_f32;
H[7] = 0.0_f32;
H[8] = 1.0_f32;
H[9] = 0.0_f32;
H[10] = 0.0_f32;
H[11] = 0.0_f32;
H[12] = 0.0_f32;
H[13] = 0.0_f32;
H[14] = 0.0_f32;
H[15] = 0.0_f32;
H[16] = 0.0_f32;
H[17] = 0.0_f32;


// Kalman gains
K[0] = ESKF_HK_POS_D1*P[0][8];
K[1] = ESKF_HK_POS_D1*P[1][8];
K[2] = ESKF_HK_POS_D1*P[2][8];
K[3] = ESKF_HK_POS_D1*P[3][8];
K[4] = ESKF_HK_POS_D1*P[4][8];
K[5] = ESKF_HK_POS_D1*P[5][8];
K[6] = ESKF_HK_POS_D1*P[6][8];
K[7] = ESKF_HK_POS_D1*P[7][8];
K[8] = ESKF_HK_POS_D1*P[8][8];
K[9] = ESKF_HK_POS_D1*P[8][9];
K[10] = ESKF_HK_POS_D1*P[8][10];
K[11] = ESKF_HK_POS_D1*P[8][11];
K[12] = ESKF_HK_POS_D1*P[8][12];
K[13] = ESKF_HK_POS_D1*P[8][13];
K[14] = ESKF_HK_POS_D1*P[8][14];
K[15] = ESKF_HK_POS_D1*P[8][15];
K[16] = ESKF_HK_POS_D1*P[8][16];
K[17] = ESKF_HK_POS_D1*P[8][17];


// Innovation Variance
S = ESKF_HK_POS_D0;
}
