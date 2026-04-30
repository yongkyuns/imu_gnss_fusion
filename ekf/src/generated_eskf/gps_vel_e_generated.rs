{
// Sub Expressions
let ESKF_HK_VEL_E0: f32 = P[4][4] + R_VEL_E;
let ESKF_HK_VEL_E1: f32 = 1.0_f32/ESKF_HK_VEL_E0;


// Observation Jacobians
H[4] = 1.0_f32;


// Kalman gains
K[0] = ESKF_HK_VEL_E1*P[0][4];
K[1] = ESKF_HK_VEL_E1*P[1][4];
K[2] = ESKF_HK_VEL_E1*P[2][4];
K[3] = ESKF_HK_VEL_E1*P[3][4];
K[4] = ESKF_HK_VEL_E1*P[4][4];
K[5] = ESKF_HK_VEL_E1*P[4][5];
K[6] = ESKF_HK_VEL_E1*P[4][6];
K[7] = ESKF_HK_VEL_E1*P[4][7];
K[8] = ESKF_HK_VEL_E1*P[4][8];
K[9] = ESKF_HK_VEL_E1*P[4][9];
K[10] = ESKF_HK_VEL_E1*P[4][10];
K[11] = ESKF_HK_VEL_E1*P[4][11];
K[12] = ESKF_HK_VEL_E1*P[4][12];
K[13] = ESKF_HK_VEL_E1*P[4][13];
K[14] = ESKF_HK_VEL_E1*P[4][14];
K[15] = ESKF_HK_VEL_E1*P[4][15];
K[16] = ESKF_HK_VEL_E1*P[4][16];
K[17] = ESKF_HK_VEL_E1*P[4][17];


// Innovation Variance
S = ESKF_HK_VEL_E0;
}
