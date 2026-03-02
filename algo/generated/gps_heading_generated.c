// Sub Expressions
const float HK0 = q0*q3 + q1*q2;
const float HK1 = 2*powf(q2, 2) + 2*powf(q3, 2) - 1;
const float HK2 = 4*powf(HK0, 2) + powf(HK1, 2);
const float HK3 = 2/HK2;
const float HK4 = HK1*q3;
const float HK5 = HK1*q2;
const float HK6 = -4*HK0*q2 + HK1*q1;
const float HK7 = -HK6;
const float HK8 = -4*HK0*q3 + HK1*q0;
const float HK9 = -HK8;
const float HK10 = HK4*P[0][0] + HK5*P[0][1] - HK7*P[0][2] - HK9*P[0][3];
const float HK11 = HK4*P[0][1] + HK5*P[1][1] - HK7*P[1][2] - HK9*P[1][3];
const float HK12 = 4/powf(HK2, 2);
const float HK13 = HK4*P[0][3] + HK5*P[1][3] - HK7*P[2][3] - HK9*P[3][3];
const float HK14 = HK4*P[0][2] + HK5*P[1][2] - HK7*P[2][2] - HK9*P[2][3];
const float HK15 = HK3/(HK10*HK12*HK4 + HK11*HK12*HK5 + HK12*HK13*HK8 + HK12*HK14*HK6 + R_YAW);


// Observation Jacobians
Hfusion[0] = -HK3*HK4;
Hfusion[1] = -HK3*HK5;
Hfusion[2] = HK3*HK7;
Hfusion[3] = HK3*HK9;
Hfusion[4] = 0;
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
Kfusion[0] = -HK10*HK15;
Kfusion[1] = -HK11*HK15;
Kfusion[2] = -HK14*HK15;
Kfusion[3] = -HK13*HK15;
Kfusion[4] = -HK15*(HK4*P[0][4] + HK5*P[1][4] - HK7*P[2][4] - HK9*P[3][4]);
Kfusion[5] = -HK15*(HK4*P[0][5] + HK5*P[1][5] - HK7*P[2][5] - HK9*P[3][5]);
Kfusion[6] = -HK15*(HK4*P[0][6] + HK5*P[1][6] - HK7*P[2][6] - HK9*P[3][6]);
Kfusion[7] = -HK15*(HK4*P[0][7] + HK5*P[1][7] - HK7*P[2][7] - HK9*P[3][7]);
Kfusion[8] = -HK15*(HK4*P[0][8] + HK5*P[1][8] - HK7*P[2][8] - HK9*P[3][8]);
Kfusion[9] = -HK15*(HK4*P[0][9] + HK5*P[1][9] - HK7*P[2][9] - HK9*P[3][9]);
Kfusion[10] = -HK15*(HK4*P[0][10] + HK5*P[1][10] - HK7*P[2][10] - HK9*P[3][10]);
Kfusion[11] = -HK15*(HK4*P[0][11] + HK5*P[1][11] - HK7*P[2][11] - HK9*P[3][11]);
Kfusion[12] = -HK15*(HK4*P[0][12] + HK5*P[1][12] - HK7*P[2][12] - HK9*P[3][12]);
Kfusion[13] = -HK15*(HK4*P[0][13] + HK5*P[1][13] - HK7*P[2][13] - HK9*P[3][13]);
Kfusion[14] = -HK15*(HK4*P[0][14] + HK5*P[1][14] - HK7*P[2][14] - HK9*P[3][14]);
Kfusion[15] = -HK15*(HK4*P[0][15] + HK5*P[1][15] - HK7*P[2][15] - HK9*P[3][15]);


