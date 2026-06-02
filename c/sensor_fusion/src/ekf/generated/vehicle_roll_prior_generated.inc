// Sub Expressions
const float tmp_hk_roll_prior0 = q0*q1 + q2*q3;
const float tmp_hk_roll_prior1 = tmp_hk_roll_prior0*tmp_hk_roll_prior0;
const float tmp_hk_roll_prior2 = q1*q1;
const float tmp_hk_roll_prior3 = q2*q2;
const float tmp_hk_roll_prior4 = 2*tmp_hk_roll_prior2 + 2*tmp_hk_roll_prior3 - 1;
const float tmp_hk_roll_prior5 = 4*tmp_hk_roll_prior1 + tmp_hk_roll_prior4*tmp_hk_roll_prior4;
const float tmp_hk_roll_prior6 = 1.0F/tmp_hk_roll_prior5;
const float tmp_hk_roll_prior7 = 4.0F*tmp_hk_roll_prior1 - 1.0F*tmp_hk_roll_prior4*(q0*q0 + q3*q3 - tmp_hk_roll_prior2 - tmp_hk_roll_prior3);
const float tmp_hk_roll_prior8 = q0*q2 - q1*q3;
const float tmp_hk_roll_prior9 = tmp_hk_roll_prior6*tmp_hk_roll_prior8;
const float tmp_hk_roll_prior10 = 4.0F*tmp_hk_roll_prior0;
const float tmp_hk_roll_prior11 = 2.0F*tmp_hk_roll_prior4;
const float tmp_hk_roll_prior12 = tmp_hk_roll_prior10*tmp_hk_roll_prior8;
const float tmp_hk_roll_prior13 = tmp_hk_roll_prior11*tmp_hk_roll_prior8;
const float tmp_hk_roll_prior14 = P[0][0]*tmp_hk_roll_prior7 + P[0][1]*tmp_hk_roll_prior12 - P[0][2]*tmp_hk_roll_prior13;
const float tmp_hk_roll_prior15 = 1.0F/(tmp_hk_roll_prior5*tmp_hk_roll_prior5);
const float tmp_hk_roll_prior16 = P[0][1]*tmp_hk_roll_prior7 + P[1][1]*tmp_hk_roll_prior12 - P[1][2]*tmp_hk_roll_prior13;
const float tmp_hk_roll_prior17 = P[0][2]*tmp_hk_roll_prior7 + P[1][2]*tmp_hk_roll_prior12 - P[2][2]*tmp_hk_roll_prior13;
const float tmp_hk_roll_prior18 = R_VEHICLE_ROLL + tmp_hk_roll_prior12*tmp_hk_roll_prior15*tmp_hk_roll_prior16 - tmp_hk_roll_prior13*tmp_hk_roll_prior15*tmp_hk_roll_prior17 + tmp_hk_roll_prior14*tmp_hk_roll_prior15*tmp_hk_roll_prior7;
const float tmp_hk_roll_prior19 = tmp_hk_roll_prior6/tmp_hk_roll_prior18;


// Observation Jacobians
H[0] = tmp_hk_roll_prior6*tmp_hk_roll_prior7;
H[1] = tmp_hk_roll_prior10*tmp_hk_roll_prior9;
H[2] = -tmp_hk_roll_prior11*tmp_hk_roll_prior9;


// Kalman gains
K[0] = tmp_hk_roll_prior14*tmp_hk_roll_prior19;
K[1] = tmp_hk_roll_prior16*tmp_hk_roll_prior19;
K[2] = tmp_hk_roll_prior17*tmp_hk_roll_prior19;
K[3] = tmp_hk_roll_prior19*(P[0][3]*tmp_hk_roll_prior7 + P[1][3]*tmp_hk_roll_prior12 - P[2][3]*tmp_hk_roll_prior13);
K[4] = tmp_hk_roll_prior19*(P[0][4]*tmp_hk_roll_prior7 + P[1][4]*tmp_hk_roll_prior12 - P[2][4]*tmp_hk_roll_prior13);
K[5] = tmp_hk_roll_prior19*(P[0][5]*tmp_hk_roll_prior7 + P[1][5]*tmp_hk_roll_prior12 - P[2][5]*tmp_hk_roll_prior13);
K[6] = tmp_hk_roll_prior19*(P[0][6]*tmp_hk_roll_prior7 + P[1][6]*tmp_hk_roll_prior12 - P[2][6]*tmp_hk_roll_prior13);
K[7] = tmp_hk_roll_prior19*(P[0][7]*tmp_hk_roll_prior7 + P[1][7]*tmp_hk_roll_prior12 - P[2][7]*tmp_hk_roll_prior13);
K[8] = tmp_hk_roll_prior19*(P[0][8]*tmp_hk_roll_prior7 + P[1][8]*tmp_hk_roll_prior12 - P[2][8]*tmp_hk_roll_prior13);
K[9] = tmp_hk_roll_prior19*(P[0][9]*tmp_hk_roll_prior7 + P[1][9]*tmp_hk_roll_prior12 - P[2][9]*tmp_hk_roll_prior13);
K[10] = tmp_hk_roll_prior19*(P[0][10]*tmp_hk_roll_prior7 + P[1][10]*tmp_hk_roll_prior12 - P[2][10]*tmp_hk_roll_prior13);
K[11] = tmp_hk_roll_prior19*(P[0][11]*tmp_hk_roll_prior7 + P[1][11]*tmp_hk_roll_prior12 - P[2][11]*tmp_hk_roll_prior13);
K[12] = tmp_hk_roll_prior19*(P[0][12]*tmp_hk_roll_prior7 + P[1][12]*tmp_hk_roll_prior12 - P[2][12]*tmp_hk_roll_prior13);
K[13] = tmp_hk_roll_prior19*(P[0][13]*tmp_hk_roll_prior7 + P[1][13]*tmp_hk_roll_prior12 - P[2][13]*tmp_hk_roll_prior13);
K[14] = tmp_hk_roll_prior19*(P[0][14]*tmp_hk_roll_prior7 + P[1][14]*tmp_hk_roll_prior12 - P[2][14]*tmp_hk_roll_prior13);
K[15] = tmp_hk_roll_prior19*(P[0][15]*tmp_hk_roll_prior7 + P[1][15]*tmp_hk_roll_prior12 - P[2][15]*tmp_hk_roll_prior13);
K[16] = tmp_hk_roll_prior19*(P[0][16]*tmp_hk_roll_prior7 + P[1][16]*tmp_hk_roll_prior12 - P[2][16]*tmp_hk_roll_prior13);
K[17] = tmp_hk_roll_prior19*(P[0][17]*tmp_hk_roll_prior7 + P[1][17]*tmp_hk_roll_prior12 - P[2][17]*tmp_hk_roll_prior13);


// Innovation Variance
S = tmp_hk_roll_prior18;
