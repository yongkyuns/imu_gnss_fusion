{
// Generated ESKF error-state noise input matrix
let ESKF_G0: f32 = -bgx*dt + dax;
let ESKF_G1: f32 = 0.5_f32*q0;
let ESKF_G2: f32 = -bgz*dt + daz;
let ESKF_G3: f32 = 0.5_f32*ESKF_G2;
let ESKF_G4: f32 = -bgy*dt + day;
let ESKF_G5: f32 = 0.5_f32*q3;
let ESKF_G6: f32 = ESKF_G0*ESKF_G1 + ESKF_G3*q2 - ESKF_G4*ESKF_G5 + q1;
let ESKF_G7: f32 = ESKF_G0*ESKF_G5 + ESKF_G1*ESKF_G4 - ESKF_G3*q1 + q2;
let ESKF_G8: f32 = 0.5_f32*q1;
let ESKF_G9: f32 = 0.5_f32*q2;
let ESKF_G10: f32 = -ESKF_G0*ESKF_G9 + ESKF_G1*ESKF_G2 + ESKF_G4*ESKF_G8 + q3;
let ESKF_G11: f32 = ESKF_G0*ESKF_G8 + ESKF_G3*q3 + ESKF_G4*ESKF_G9 - q0;
let ESKF_G12: f32 = 1.0_f32*ESKF_G10*q3 - 1.0_f32*ESKF_G11*q0 + 1.0_f32*ESKF_G6*q1 + 1.0_f32*ESKF_G7*q2;
let ESKF_G13: f32 = ESKF_G10*q0 + ESKF_G11*q3 + ESKF_G6*q2 - ESKF_G7*q1;
let ESKF_G14: f32 = ESKF_G10*q1 + ESKF_G11*q2 - ESKF_G6*q3 + ESKF_G7*q0;
let ESKF_G15: f32 = -ESKF_G10*q2 + ESKF_G11*q1 + ESKF_G6*q0 + ESKF_G7*q3;
let ESKF_G16: f32 = 2.0_f32*q2*q2;
let ESKF_G17: f32 = 2.0_f32*q3*q3 - 1.0_f32;
let ESKF_G18: f32 = q0*q3;
let ESKF_G19: f32 = q0*q2;
let ESKF_G20: f32 = 2.0_f32*q1*q1;
let ESKF_G21: f32 = q0*q1;
let ESKF_G22: f32 = 1.0_f32*(qcs0*qcs0 + qcs1*qcs1 + qcs2*qcs2 + qcs3*qcs3);


G[0][0] = ESKF_G12;
G[1][0] = -1.0_f32*ESKF_G13;
G[2][0] = 1.0_f32*ESKF_G14;
G[0][1] = 1.0_f32*ESKF_G13;
G[1][1] = ESKF_G12;
G[2][1] = -1.0_f32*ESKF_G15;
G[0][2] = -1.0_f32*ESKF_G14;
G[1][2] = 1.0_f32*ESKF_G15;
G[2][2] = ESKF_G12;
G[3][3] = -ESKF_G16 - ESKF_G17;
G[4][3] = 2.0_f32*ESKF_G18 + 2.0_f32*q1*q2;
G[5][3] = -2.0_f32*ESKF_G19 + 2.0_f32*q1*q3;
G[3][4] = -2.0_f32*ESKF_G18 + 2.0_f32*q1*q2;
G[4][4] = -ESKF_G17 - ESKF_G20;
G[5][4] = 2.0_f32*ESKF_G21 + 2.0_f32*q2*q3;
G[3][5] = 2.0_f32*ESKF_G19 + 2.0_f32*q1*q3;
G[4][5] = -2.0_f32*ESKF_G21 + 2.0_f32*q2*q3;
G[5][5] = -ESKF_G16 - ESKF_G20 + 1.0_f32;
G[9][6] = dt;
G[10][7] = dt;
G[11][8] = dt;
G[12][9] = dt;
G[13][10] = dt;
G[14][11] = dt;
G[15][12] = ESKF_G22;
G[16][13] = ESKF_G22;
G[17][14] = ESKF_G22;


}
