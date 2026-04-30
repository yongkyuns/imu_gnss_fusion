{
// Sub Expressions
let tmp_G_0: f32 = 2.0_f32*q2*q2;
let tmp_G_1: f32 = 2.0_f32*q3*q3 - 1.0_f32;
let tmp_G_2: f32 = -tmp_G_0 - tmp_G_1;
let tmp_G_3: f32 = q0*q3;
let tmp_G_4: f32 = 2.0_f32*q1*q2 - 2.0_f32*tmp_G_3;
let tmp_G_5: f32 = q0*q2;
let tmp_G_6: f32 = 2.0_f32*q1*q3 + 2.0_f32*tmp_G_5;
let tmp_G_7: f32 = 2.0_f32*q1*q2 + 2.0_f32*tmp_G_3;
let tmp_G_8: f32 = 2.0_f32*q1*q1;
let tmp_G_9: f32 = -tmp_G_1 - tmp_G_8;
let tmp_G_10: f32 = q0*q1;
let tmp_G_11: f32 = 2.0_f32*q2*q3 - 2.0_f32*tmp_G_10;
let tmp_G_12: f32 = 2.0_f32*q1*q3 - 2.0_f32*tmp_G_5;
let tmp_G_13: f32 = 2.0_f32*q2*q3 + 2.0_f32*tmp_G_10;
let tmp_G_14: f32 = -tmp_G_0 - tmp_G_8 + 1.0_f32;


// G
G[3][0] = tmp_G_2;
G[4][0] = tmp_G_7;
G[5][0] = tmp_G_12;
G[3][1] = tmp_G_4;
G[4][1] = tmp_G_9;
G[5][1] = tmp_G_13;
G[3][2] = tmp_G_6;
G[4][2] = tmp_G_11;
G[5][2] = tmp_G_14;
G[6][3] = tmp_G_2;
G[7][3] = tmp_G_7;
G[8][3] = tmp_G_12;
G[6][4] = tmp_G_4;
G[7][4] = tmp_G_9;
G[8][4] = tmp_G_13;
G[6][5] = tmp_G_6;
G[7][5] = tmp_G_11;
G[8][5] = tmp_G_14;
G[9][6] = 1.0_f32;
G[10][7] = 1.0_f32;
G[11][8] = 1.0_f32;
G[12][9] = 1.0_f32;
G[13][10] = 1.0_f32;
G[14][11] = 1.0_f32;
G[15][12] = 1.0_f32;
G[16][13] = 1.0_f32;
G[17][14] = 1.0_f32;
G[18][15] = 1.0_f32;
G[19][16] = 1.0_f32;
G[20][17] = 1.0_f32;
G[21][18] = 1.0_f32;
G[22][19] = 1.0_f32;
G[23][20] = 1.0_f32;


}
