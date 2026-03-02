// Sub Expressions
const float HK0 = 1.0F/(P[9][9] + R_POS_D);


// Observation Jacobians
Hfusion[0] = 0;
Hfusion[1] = 0;
Hfusion[2] = 0;
Hfusion[3] = 0;
Hfusion[4] = 0;
Hfusion[5] = 0;
Hfusion[6] = 0;
Hfusion[7] = 0;
Hfusion[8] = 0;
Hfusion[9] = 1;
Hfusion[10] = 0;
Hfusion[11] = 0;
Hfusion[12] = 0;
Hfusion[13] = 0;
Hfusion[14] = 0;
Hfusion[15] = 0;


// Kalman gains
Kfusion[0] = HK0*P[0][9];
Kfusion[1] = HK0*P[1][9];
Kfusion[2] = HK0*P[2][9];
Kfusion[3] = HK0*P[3][9];
Kfusion[4] = HK0*P[4][9];
Kfusion[5] = HK0*P[5][9];
Kfusion[6] = HK0*P[6][9];
Kfusion[7] = HK0*P[7][9];
Kfusion[8] = HK0*P[8][9];
Kfusion[9] = HK0*P[9][9];
Kfusion[10] = HK0*P[9][10];
Kfusion[11] = HK0*P[9][11];
Kfusion[12] = HK0*P[9][12];
Kfusion[13] = HK0*P[9][13];
Kfusion[14] = HK0*P[9][14];
Kfusion[15] = HK0*P[9][15];


