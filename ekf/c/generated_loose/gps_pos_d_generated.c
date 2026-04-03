// Sub Expressions
const float LOOSE_HK_POS_D0 = P[8][8] + R_POS_D;
const float LOOSE_HK_POS_D1 = 1.0F/LOOSE_HK_POS_D0;


// Observation Jacobians
H[0] = 0;
H[1] = 0;
H[2] = 0;
H[3] = 0;
H[4] = 0;
H[5] = 0;
H[6] = 0;
H[7] = 0;
H[8] = 1;
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
K[0] = LOOSE_HK_POS_D1*P[0][8];
K[1] = LOOSE_HK_POS_D1*P[1][8];
K[2] = LOOSE_HK_POS_D1*P[2][8];
K[3] = LOOSE_HK_POS_D1*P[3][8];
K[4] = LOOSE_HK_POS_D1*P[4][8];
K[5] = LOOSE_HK_POS_D1*P[5][8];
K[6] = LOOSE_HK_POS_D1*P[6][8];
K[7] = LOOSE_HK_POS_D1*P[7][8];
K[8] = LOOSE_HK_POS_D1*P[8][8];
K[9] = LOOSE_HK_POS_D1*P[8][9];
K[10] = LOOSE_HK_POS_D1*P[8][10];
K[11] = LOOSE_HK_POS_D1*P[8][11];
K[12] = LOOSE_HK_POS_D1*P[8][12];
K[13] = LOOSE_HK_POS_D1*P[8][13];
K[14] = LOOSE_HK_POS_D1*P[8][14];
K[15] = LOOSE_HK_POS_D1*P[8][15];
K[16] = LOOSE_HK_POS_D1*P[8][16];
K[17] = LOOSE_HK_POS_D1*P[8][17];
K[18] = LOOSE_HK_POS_D1*P[8][18];
K[19] = LOOSE_HK_POS_D1*P[8][19];
K[20] = LOOSE_HK_POS_D1*P[8][20];
K[21] = LOOSE_HK_POS_D1*P[8][21];
K[22] = LOOSE_HK_POS_D1*P[8][22];
K[23] = LOOSE_HK_POS_D1*P[8][23];


// Innovation Variance
S = LOOSE_HK_POS_D0;
