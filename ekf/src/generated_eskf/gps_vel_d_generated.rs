{
// Sub Expressions
let ESKF_HK_VEL_D0: f32 = P[5][5] + R_VEL_D;
let ESKF_HK_VEL_D1: f32 = 1.0_f32/ESKF_HK_VEL_D0;


// Observation Jacobians
H[5] = 1.0_f32;


// Kalman gains
K[0] = ESKF_HK_VEL_D1*P[0][5];
K[1] = ESKF_HK_VEL_D1*P[1][5];
K[2] = ESKF_HK_VEL_D1*P[2][5];
K[3] = ESKF_HK_VEL_D1*P[3][5];
K[4] = ESKF_HK_VEL_D1*P[4][5];
K[5] = ESKF_HK_VEL_D1*P[5][5];
K[6] = ESKF_HK_VEL_D1*P[5][6];
K[7] = ESKF_HK_VEL_D1*P[5][7];
K[8] = ESKF_HK_VEL_D1*P[5][8];
K[9] = ESKF_HK_VEL_D1*P[5][9];
K[10] = ESKF_HK_VEL_D1*P[5][10];
K[11] = ESKF_HK_VEL_D1*P[5][11];
K[12] = ESKF_HK_VEL_D1*P[5][12];
K[13] = ESKF_HK_VEL_D1*P[5][13];
K[14] = ESKF_HK_VEL_D1*P[5][14];
K[15] = ESKF_HK_VEL_D1*P[5][15];
K[16] = ESKF_HK_VEL_D1*P[5][16];
K[17] = ESKF_HK_VEL_D1*P[5][17];


// Innovation Variance
S = ESKF_HK_VEL_D0;
}
