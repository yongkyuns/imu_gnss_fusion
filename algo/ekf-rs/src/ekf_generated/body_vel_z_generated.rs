{
// Sub Expressions
let HK0 = q1*ve - q2*vn;
let HK1 = 2.0*vd;
let HK2 = HK1*q1 + q0*ve - q3*vn;
let HK3 = -HK1*q2 + q0*vn + q3*ve;
let HK4 = q1*vn + q2*ve;
let HK5 = q0*q2 + q1*q3;
let HK6 = q0*q1 - q2*q3;
let HK7 = 2.0*powf(q1, 2.0) + 2.0*powf(q2, 2.0) - 1.0;
let HK8 = 2.0*HK4;
let HK9 = 2.0*HK5;
let HK10 = 2.0*HK0;
let HK11 = 2.0*HK6;
let HK12 = 2.0*HK3;
let HK13 = 2.0*HK2;
let HK14 = HK10*P[0][0] + HK11*P[0][5] - HK12*P[0][2] + HK13*P[0][1] + HK7*P[0][6] - HK8*P[0][3] - HK9*P[0][4];
let HK15 = HK10*P[0][4] + HK11*P[4][5] - HK12*P[2][4] + HK13*P[1][4] + HK7*P[4][6] - HK8*P[3][4] - HK9*P[4][4];
let HK16 = HK10*P[0][3] + HK11*P[3][5] - HK12*P[2][3] + HK13*P[1][3] + HK7*P[3][6] - HK8*P[3][3] - HK9*P[3][4];
let HK17 = HK10*P[0][5] + HK11*P[5][5] - HK12*P[2][5] + HK13*P[1][5] + HK7*P[5][6] - HK8*P[3][5] - HK9*P[4][5];
let HK18 = HK10*P[0][6] + HK11*P[5][6] - HK12*P[2][6] + HK13*P[1][6] + HK7*P[6][6] - HK8*P[3][6] - HK9*P[4][6];
let HK19 = HK10*P[0][2] + HK11*P[2][5] - HK12*P[2][2] + HK13*P[1][2] + HK7*P[2][6] - HK8*P[2][3] - HK9*P[2][4];
let HK20 = HK10*P[0][1] + HK11*P[1][5] - HK12*P[1][2] + HK13*P[1][1] + HK7*P[1][6] - HK8*P[1][3] - HK9*P[1][4];
let HK21 = 1.0/(HK10*HK14 + HK11*HK17 - HK12*HK19 + HK13*HK20 - HK15*HK9 - HK16*HK8 + HK18*HK7 + R_BODY_VEL);


// Observation Jacobians
Hfusion[0] = -2.0*HK0;
Hfusion[1] = -2.0*HK2;
Hfusion[2] = 2.0*HK3;
Hfusion[3] = 2.0*HK4;
Hfusion[4] = 2.0*HK5;
Hfusion[5] = -2.0*HK6;
Hfusion[6] = -HK7;
Hfusion[7] = 0.0;
Hfusion[8] = 0.0;
Hfusion[9] = 0.0;
Hfusion[10] = 0.0;
Hfusion[11] = 0.0;
Hfusion[12] = 0.0;
Hfusion[13] = 0.0;
Hfusion[14] = 0.0;
Hfusion[15] = 0.0;


// Kalman gains
Kfusion[0] = -HK14*HK21;
Kfusion[1] = -HK20*HK21;
Kfusion[2] = -HK19*HK21;
Kfusion[3] = -HK16*HK21;
Kfusion[4] = -HK15*HK21;
Kfusion[5] = -HK17*HK21;
Kfusion[6] = -HK18*HK21;
Kfusion[7] = -HK21*(HK10*P[0][7] + HK11*P[5][7] - HK12*P[2][7] + HK13*P[1][7] + HK7*P[6][7] - HK8*P[3][7] - HK9*P[4][7]);
Kfusion[8] = -HK21*(HK10*P[0][8] + HK11*P[5][8] - HK12*P[2][8] + HK13*P[1][8] + HK7*P[6][8] - HK8*P[3][8] - HK9*P[4][8]);
Kfusion[9] = -HK21*(HK10*P[0][9] + HK11*P[5][9] - HK12*P[2][9] + HK13*P[1][9] + HK7*P[6][9] - HK8*P[3][9] - HK9*P[4][9]);
Kfusion[10] = -HK21*(HK10*P[0][10] + HK11*P[5][10] - HK12*P[2][10] + HK13*P[1][10] + HK7*P[6][10] - HK8*P[3][10] - HK9*P[4][10]);
Kfusion[11] = -HK21*(HK10*P[0][11] + HK11*P[5][11] - HK12*P[2][11] + HK13*P[1][11] + HK7*P[6][11] - HK8*P[3][11] - HK9*P[4][11]);
Kfusion[12] = -HK21*(HK10*P[0][12] + HK11*P[5][12] - HK12*P[2][12] + HK13*P[1][12] + HK7*P[6][12] - HK8*P[3][12] - HK9*P[4][12]);
Kfusion[13] = -HK21*(HK10*P[0][13] + HK11*P[5][13] - HK12*P[2][13] + HK13*P[1][13] + HK7*P[6][13] - HK8*P[3][13] - HK9*P[4][13]);
Kfusion[14] = -HK21*(HK10*P[0][14] + HK11*P[5][14] - HK12*P[2][14] + HK13*P[1][14] + HK7*P[6][14] - HK8*P[3][14] - HK9*P[4][14]);
Kfusion[15] = -HK21*(HK10*P[0][15] + HK11*P[5][15] - HK12*P[2][15] + HK13*P[1][15] + HK7*P[6][15] - HK8*P[3][15] - HK9*P[4][15]);


}
