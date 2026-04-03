// Sub Expressions
const float LOOSE_HK_POS_E0 = P[7][7] + R_POS_E;
const float LOOSE_HK_POS_E1 = 1.0F/LOOSE_HK_POS_E0;


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
H[18] = 0;
H[19] = 0;
H[20] = 0;
H[21] = 0;
H[22] = 0;
H[23] = 0;


// Kalman gains
K[0] = LOOSE_HK_POS_E1*P[0][7];
K[1] = LOOSE_HK_POS_E1*P[1][7];
K[2] = LOOSE_HK_POS_E1*P[2][7];
K[3] = LOOSE_HK_POS_E1*P[3][7];
K[4] = LOOSE_HK_POS_E1*P[4][7];
K[5] = LOOSE_HK_POS_E1*P[5][7];
K[6] = LOOSE_HK_POS_E1*P[6][7];
K[7] = LOOSE_HK_POS_E1*P[7][7];
K[8] = LOOSE_HK_POS_E1*P[7][8];
K[9] = LOOSE_HK_POS_E1*P[7][9];
K[10] = LOOSE_HK_POS_E1*P[7][10];
K[11] = LOOSE_HK_POS_E1*P[7][11];
K[12] = LOOSE_HK_POS_E1*P[7][12];
K[13] = LOOSE_HK_POS_E1*P[7][13];
K[14] = LOOSE_HK_POS_E1*P[7][14];
K[15] = LOOSE_HK_POS_E1*P[7][15];
K[16] = LOOSE_HK_POS_E1*P[7][16];
K[17] = LOOSE_HK_POS_E1*P[7][17];
K[18] = LOOSE_HK_POS_E1*P[7][18];
K[19] = LOOSE_HK_POS_E1*P[7][19];
K[20] = LOOSE_HK_POS_E1*P[7][20];
K[21] = LOOSE_HK_POS_E1*P[7][21];
K[22] = LOOSE_HK_POS_E1*P[7][22];
K[23] = LOOSE_HK_POS_E1*P[7][23];


// Innovation Variance
S = LOOSE_HK_POS_E0;
