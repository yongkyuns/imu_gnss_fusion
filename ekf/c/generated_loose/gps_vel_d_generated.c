// Sub Expressions
const float LOOSE_HK_VEL_D0 = P[5][5] + R_VEL_D;
const float LOOSE_HK_VEL_D1 = 1.0F/LOOSE_HK_VEL_D0;


// Observation Jacobians
H[0] = 0;
H[1] = 0;
H[2] = 0;
H[3] = 0;
H[4] = 0;
H[5] = 1;
H[6] = 0;
H[7] = 0;
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
K[0] = LOOSE_HK_VEL_D1*P[0][5];
K[1] = LOOSE_HK_VEL_D1*P[1][5];
K[2] = LOOSE_HK_VEL_D1*P[2][5];
K[3] = LOOSE_HK_VEL_D1*P[3][5];
K[4] = LOOSE_HK_VEL_D1*P[4][5];
K[5] = LOOSE_HK_VEL_D1*P[5][5];
K[6] = LOOSE_HK_VEL_D1*P[5][6];
K[7] = LOOSE_HK_VEL_D1*P[5][7];
K[8] = LOOSE_HK_VEL_D1*P[5][8];
K[9] = LOOSE_HK_VEL_D1*P[5][9];
K[10] = LOOSE_HK_VEL_D1*P[5][10];
K[11] = LOOSE_HK_VEL_D1*P[5][11];
K[12] = LOOSE_HK_VEL_D1*P[5][12];
K[13] = LOOSE_HK_VEL_D1*P[5][13];
K[14] = LOOSE_HK_VEL_D1*P[5][14];
K[15] = LOOSE_HK_VEL_D1*P[5][15];
K[16] = LOOSE_HK_VEL_D1*P[5][16];
K[17] = LOOSE_HK_VEL_D1*P[5][17];
K[18] = LOOSE_HK_VEL_D1*P[5][18];
K[19] = LOOSE_HK_VEL_D1*P[5][19];
K[20] = LOOSE_HK_VEL_D1*P[5][20];
K[21] = LOOSE_HK_VEL_D1*P[5][21];
K[22] = LOOSE_HK_VEL_D1*P[5][22];
K[23] = LOOSE_HK_VEL_D1*P[5][23];


// Innovation Variance
S = LOOSE_HK_VEL_D0;
