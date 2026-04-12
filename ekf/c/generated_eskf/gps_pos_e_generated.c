// Sub Expressions
const float ESKF_HK_POS_E0 = P[7][7] + R_POS_E;
const float ESKF_HK_POS_E1 = 1.0F/ESKF_HK_POS_E0;


// Observation Jacobians
H[0] = 0;
H[1] = 0;
H[2] = 0;
H[3] = 0;
H[4] = 0;
H[5] = 0;
H[6] = 0;
H[7] = 1;
H[8] = 0;
H[9] = 0;
H[10] = 0;
H[11] = 0;
H[12] = 0;
H[13] = 0;
H[14] = 0;
H[15] = 0;
H[16] = 0;
H[17] = 0;


// Kalman gains
K[0] = ESKF_HK_POS_E1*P[0][7];
K[1] = ESKF_HK_POS_E1*P[1][7];
K[2] = ESKF_HK_POS_E1*P[2][7];
K[3] = ESKF_HK_POS_E1*P[3][7];
K[4] = ESKF_HK_POS_E1*P[4][7];
K[5] = ESKF_HK_POS_E1*P[5][7];
K[6] = ESKF_HK_POS_E1*P[6][7];
K[7] = ESKF_HK_POS_E1*P[7][7];
K[8] = ESKF_HK_POS_E1*P[7][8];
K[9] = ESKF_HK_POS_E1*P[7][9];
K[10] = ESKF_HK_POS_E1*P[7][10];
K[11] = ESKF_HK_POS_E1*P[7][11];
K[12] = ESKF_HK_POS_E1*P[7][12];
K[13] = ESKF_HK_POS_E1*P[7][13];
K[14] = ESKF_HK_POS_E1*P[7][14];
K[15] = ESKF_HK_POS_E1*P[7][15];
K[16] = ESKF_HK_POS_E1*P[7][16];
K[17] = ESKF_HK_POS_E1*P[7][17];


// Innovation Variance
S = ESKF_HK_POS_E0;
