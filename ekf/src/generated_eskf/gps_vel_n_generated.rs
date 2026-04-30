{
// Sub Expressions
let ESKF_HK_VEL_N0: f32 = P[3][3] + R_VEL_N;
let ESKF_HK_VEL_N1: f32 = 1.0_f32/ESKF_HK_VEL_N0;


// Observation Jacobians
H[3] = 1.0_f32;


// Kalman gains
K[0] = ESKF_HK_VEL_N1*P[0][3];
K[1] = ESKF_HK_VEL_N1*P[1][3];
K[2] = ESKF_HK_VEL_N1*P[2][3];
K[3] = ESKF_HK_VEL_N1*P[3][3];
K[4] = ESKF_HK_VEL_N1*P[3][4];
K[5] = ESKF_HK_VEL_N1*P[3][5];
K[6] = ESKF_HK_VEL_N1*P[3][6];
K[7] = ESKF_HK_VEL_N1*P[3][7];
K[8] = ESKF_HK_VEL_N1*P[3][8];
K[9] = ESKF_HK_VEL_N1*P[3][9];
K[10] = ESKF_HK_VEL_N1*P[3][10];
K[11] = ESKF_HK_VEL_N1*P[3][11];
K[12] = ESKF_HK_VEL_N1*P[3][12];
K[13] = ESKF_HK_VEL_N1*P[3][13];
K[14] = ESKF_HK_VEL_N1*P[3][14];
K[15] = ESKF_HK_VEL_N1*P[3][15];
K[16] = ESKF_HK_VEL_N1*P[3][16];
K[17] = ESKF_HK_VEL_N1*P[3][17];


// Innovation Variance
S = ESKF_HK_VEL_N0;
}
