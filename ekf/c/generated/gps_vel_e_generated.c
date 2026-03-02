// Sub Expressions
const float HK0 = 1.0F/(P[5][5] + R_VEL_E);


// Observation Jacobians
Hfusion[0] = 0;
Hfusion[1] = 0;
Hfusion[2] = 0;
Hfusion[3] = 0;
Hfusion[4] = 0;
Hfusion[5] = 1;
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
Kfusion[0] = HK0*P[0][5];
Kfusion[1] = HK0*P[1][5];
Kfusion[2] = HK0*P[2][5];
Kfusion[3] = HK0*P[3][5];
Kfusion[4] = HK0*P[4][5];
Kfusion[5] = HK0*P[5][5];
Kfusion[6] = HK0*P[5][6];
Kfusion[7] = HK0*P[5][7];
Kfusion[8] = HK0*P[5][8];
Kfusion[9] = HK0*P[5][9];
Kfusion[10] = HK0*P[5][10];
Kfusion[11] = HK0*P[5][11];
Kfusion[12] = HK0*P[5][12];
Kfusion[13] = HK0*P[5][13];
Kfusion[14] = HK0*P[5][14];
Kfusion[15] = HK0*P[5][15];


