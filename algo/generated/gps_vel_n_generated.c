// Sub Expressions
const float HK0 = 1.0F/(P[4][4] + R_VEL_N);


// Observation Jacobians
Hfusion[0] = 0;
Hfusion[1] = 0;
Hfusion[2] = 0;
Hfusion[3] = 0;
Hfusion[4] = 1;
Hfusion[5] = 0;
Hfusion[6] = 0;
Hfusion[7] = 0;
Hfusion[8] = 0;
Hfusion[9] = 0;
Hfusion[10] = 0;
Hfusion[11] = 0;
Hfusion[12] = 0;
Hfusion[13] = 0;
Hfusion[14] = 0;
Hfusion[15] = 0;


// Kalman gains
Kfusion[0] = HK0*P[0][4];
Kfusion[1] = HK0*P[1][4];
Kfusion[2] = HK0*P[2][4];
Kfusion[3] = HK0*P[3][4];
Kfusion[4] = HK0*P[4][4];
Kfusion[5] = HK0*P[4][5];
Kfusion[6] = HK0*P[4][6];
Kfusion[7] = HK0*P[4][7];
Kfusion[8] = HK0*P[4][8];
Kfusion[9] = HK0*P[4][9];
Kfusion[10] = HK0*P[4][10];
Kfusion[11] = HK0*P[4][11];
Kfusion[12] = HK0*P[4][12];
Kfusion[13] = HK0*P[4][13];
Kfusion[14] = HK0*P[4][14];
Kfusion[15] = HK0*P[4][15];


