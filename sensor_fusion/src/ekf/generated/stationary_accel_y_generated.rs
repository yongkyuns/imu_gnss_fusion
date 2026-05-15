{
// Sub Expressions
let tmp_hk_stat_ay0: f32 = 1.0_f32*q0*q0 - 1.0_f32*q1*q1 - 1.0_f32*q2*q2 + 1.0_f32*q3*q3;
let tmp_hk_stat_ay1: f32 = 2.0_f32*q0*q2 - 2.0_f32*q1*q3;
let tmp_hk_stat_ay2: f32 = P[0][0]*tmp_hk_stat_ay0 + P[0][2]*tmp_hk_stat_ay1;
let tmp_hk_stat_ay3: f32 = P[0][2]*tmp_hk_stat_ay0 + P[2][2]*tmp_hk_stat_ay1;
let tmp_hk_stat_ay4: f32 = g*g;
let tmp_hk_stat_ay5: f32 = R_STATIONARY_ACCEL + tmp_hk_stat_ay0*tmp_hk_stat_ay2*tmp_hk_stat_ay4 + tmp_hk_stat_ay1*tmp_hk_stat_ay3*tmp_hk_stat_ay4;
let tmp_hk_stat_ay6: f32 = g/tmp_hk_stat_ay5;


// Observation Jacobians
H[0] = -g*tmp_hk_stat_ay0;
H[2] = -g*tmp_hk_stat_ay1;


// Kalman gains
K[0] = -tmp_hk_stat_ay2*tmp_hk_stat_ay6;
K[1] = -tmp_hk_stat_ay6*(P[0][1]*tmp_hk_stat_ay0 + P[1][2]*tmp_hk_stat_ay1);
K[2] = -tmp_hk_stat_ay3*tmp_hk_stat_ay6;
K[3] = -tmp_hk_stat_ay6*(P[0][3]*tmp_hk_stat_ay0 + P[2][3]*tmp_hk_stat_ay1);
K[4] = -tmp_hk_stat_ay6*(P[0][4]*tmp_hk_stat_ay0 + P[2][4]*tmp_hk_stat_ay1);
K[5] = -tmp_hk_stat_ay6*(P[0][5]*tmp_hk_stat_ay0 + P[2][5]*tmp_hk_stat_ay1);
K[6] = -tmp_hk_stat_ay6*(P[0][6]*tmp_hk_stat_ay0 + P[2][6]*tmp_hk_stat_ay1);
K[7] = -tmp_hk_stat_ay6*(P[0][7]*tmp_hk_stat_ay0 + P[2][7]*tmp_hk_stat_ay1);
K[8] = -tmp_hk_stat_ay6*(P[0][8]*tmp_hk_stat_ay0 + P[2][8]*tmp_hk_stat_ay1);
K[9] = -tmp_hk_stat_ay6*(P[0][9]*tmp_hk_stat_ay0 + P[2][9]*tmp_hk_stat_ay1);
K[10] = -tmp_hk_stat_ay6*(P[0][10]*tmp_hk_stat_ay0 + P[2][10]*tmp_hk_stat_ay1);
K[11] = -tmp_hk_stat_ay6*(P[0][11]*tmp_hk_stat_ay0 + P[2][11]*tmp_hk_stat_ay1);
K[12] = -tmp_hk_stat_ay6*(P[0][12]*tmp_hk_stat_ay0 + P[2][12]*tmp_hk_stat_ay1);
K[13] = -tmp_hk_stat_ay6*(P[0][13]*tmp_hk_stat_ay0 + P[2][13]*tmp_hk_stat_ay1);
K[14] = -tmp_hk_stat_ay6*(P[0][14]*tmp_hk_stat_ay0 + P[2][14]*tmp_hk_stat_ay1);
K[15] = -tmp_hk_stat_ay6*(P[0][15]*tmp_hk_stat_ay0 + P[2][15]*tmp_hk_stat_ay1);
K[16] = -tmp_hk_stat_ay6*(P[0][16]*tmp_hk_stat_ay0 + P[2][16]*tmp_hk_stat_ay1);
K[17] = -tmp_hk_stat_ay6*(P[0][17]*tmp_hk_stat_ay0 + P[2][17]*tmp_hk_stat_ay1);


// Innovation Variance
S = tmp_hk_stat_ay5;
}
