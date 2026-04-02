// Sub Expressions
const float ESKF_HK_POS_D0 = P[8][8] + R_POS_D;
const float ESKF_HK_POS_D1 = 1.0F/ESKF_HK_POS_D0;


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


// Innovation Variance
S = ESKF_HK_POS_D0;
