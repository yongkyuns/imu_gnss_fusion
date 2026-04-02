// Sub Expressions
const float ESKF_HK_POS_N0 = P[6][6] + R_POS_N;
const float ESKF_HK_POS_N1 = 1.0F/ESKF_HK_POS_N0;


// Observation Jacobians
H[0] = 0;
H[1] = 0;
H[2] = 0;
H[3] = 0;
H[4] = 0;
H[5] = 0;
H[6] = 1;
H[7] = 0;
H[8] = 0;
H[9] = 0;
H[10] = 0;
H[11] = 0;
H[12] = 0;
H[13] = 0;
H[14] = 0;


// Kalman gains
K[0] = ESKF_HK_POS_N1*P[0][6];
K[1] = ESKF_HK_POS_N1*P[1][6];
K[2] = ESKF_HK_POS_N1*P[2][6];
K[3] = ESKF_HK_POS_N1*P[3][6];
K[4] = ESKF_HK_POS_N1*P[4][6];
K[5] = ESKF_HK_POS_N1*P[5][6];
K[6] = ESKF_HK_POS_N1*P[6][6];
K[7] = ESKF_HK_POS_N1*P[6][7];
K[8] = ESKF_HK_POS_N1*P[6][8];
K[9] = ESKF_HK_POS_N1*P[6][9];
K[10] = ESKF_HK_POS_N1*P[6][10];
K[11] = ESKF_HK_POS_N1*P[6][11];
K[12] = ESKF_HK_POS_N1*P[6][12];
K[13] = ESKF_HK_POS_N1*P[6][13];
K[14] = ESKF_HK_POS_N1*P[6][14];


// Innovation Variance
S = ESKF_HK_POS_N0;
