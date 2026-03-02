{
// Sub Expressions
let HK0 = 1.0/(P[7][7] + R_POS_N);


// Observation Jacobians
Hfusion[0] = 0.0;
Hfusion[1] = 0.0;
Hfusion[2] = 0.0;
Hfusion[3] = 0.0;
Hfusion[4] = 0.0;
Hfusion[5] = 0.0;
Hfusion[6] = 0.0;
Hfusion[7] = 1.0;
Hfusion[8] = 0.0;
Hfusion[9] = 0.0;
Hfusion[10] = 0.0;
Hfusion[11] = 0.0;
Hfusion[12] = 0.0;
Hfusion[13] = 0.0;
Hfusion[14] = 0.0;
Hfusion[15] = 0.0;


// Kalman gains
Kfusion[0] = HK0*P[0][7];
Kfusion[1] = HK0*P[1][7];
Kfusion[2] = HK0*P[2][7];
Kfusion[3] = HK0*P[3][7];
Kfusion[4] = HK0*P[4][7];
Kfusion[5] = HK0*P[5][7];
Kfusion[6] = HK0*P[6][7];
Kfusion[7] = HK0*P[7][7];
Kfusion[8] = HK0*P[7][8];
Kfusion[9] = HK0*P[7][9];
Kfusion[10] = HK0*P[7][10];
Kfusion[11] = HK0*P[7][11];
Kfusion[12] = HK0*P[7][12];
Kfusion[13] = HK0*P[7][13];
Kfusion[14] = HK0*P[7][14];
Kfusion[15] = HK0*P[7][15];


}
