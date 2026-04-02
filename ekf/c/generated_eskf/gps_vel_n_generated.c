// Sub Expressions
const float ESKF_HK_VEL_N0 = 1.0F/(P[3][3] + R_VEL_N);


// Observation Jacobians
H[0] = 0;
H[1] = 0;
H[2] = 0;
H[3] = 1;
H[4] = 0;
H[5] = 0;
H[6] = 0;
H[7] = 0;
H[8] = 0;
H[9] = 0;
H[10] = 0;
H[11] = 0;
H[12] = 0;
H[13] = 0;
H[14] = 0;


// Kalman gains
K[0] = ESKF_HK_VEL_N0*P[0][3];
K[1] = ESKF_HK_VEL_N0*P[1][3];
K[2] = ESKF_HK_VEL_N0*P[2][3];
K[3] = ESKF_HK_VEL_N0*P[3][3];
K[4] = ESKF_HK_VEL_N0*P[3][4];
K[5] = ESKF_HK_VEL_N0*P[3][5];
K[6] = ESKF_HK_VEL_N0*P[3][6];
K[7] = ESKF_HK_VEL_N0*P[3][7];
K[8] = ESKF_HK_VEL_N0*P[3][8];
K[9] = ESKF_HK_VEL_N0*P[3][9];
K[10] = ESKF_HK_VEL_N0*P[3][10];
K[11] = ESKF_HK_VEL_N0*P[3][11];
K[12] = ESKF_HK_VEL_N0*P[3][12];
K[13] = ESKF_HK_VEL_N0*P[3][13];
K[14] = ESKF_HK_VEL_N0*P[3][14];


