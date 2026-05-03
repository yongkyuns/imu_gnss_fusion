{
// Sub Expressions
let tmp_G_0: f32 = q0*q2;
let tmp_G_1: f32 = q1*q3;
let tmp_G_2: f32 = tmp_G_0 + tmp_G_1;
let tmp_G_3: f32 = qcs0*qcs2;
let tmp_G_4: f32 = qcs1*qcs3;
let tmp_G_5: f32 = tmp_G_3 + tmp_G_4;
let tmp_G_6: f32 = q0*q3;
let tmp_G_7: f32 = q1*q2;
let tmp_G_8: f32 = tmp_G_6 - tmp_G_7;
let tmp_G_9: f32 = qcs0*qcs3;
let tmp_G_10: f32 = qcs1*qcs2;
let tmp_G_11: f32 = -tmp_G_10 + tmp_G_9;
let tmp_G_12: f32 = 2.0_f32*q2*q2;
let tmp_G_13: f32 = 2.0_f32*q3*q3 - 1.0_f32;
let tmp_G_14: f32 = tmp_G_12 + tmp_G_13;
let tmp_G_15: f32 = 2.0_f32*qcs2*qcs2;
let tmp_G_16: f32 = 2.0_f32*qcs3*qcs3 - 1.0_f32;
let tmp_G_17: f32 = tmp_G_15 + tmp_G_16;
let tmp_G_18: f32 = 4.0_f32*tmp_G_11*tmp_G_8 + tmp_G_14*tmp_G_17 + 4.0_f32*tmp_G_2*tmp_G_5;
let tmp_G_19: f32 = qcs0*qcs1;
let tmp_G_20: f32 = qcs2*qcs3;
let tmp_G_21: f32 = tmp_G_19 - tmp_G_20;
let tmp_G_22: f32 = tmp_G_10 + tmp_G_9;
let tmp_G_23: f32 = 2.0_f32*qcs1*qcs1;
let tmp_G_24: f32 = tmp_G_16 + tmp_G_23;
let tmp_G_25: f32 = -2.0_f32*tmp_G_14*tmp_G_22 - 4.0_f32*tmp_G_2*tmp_G_21 + 2.0_f32*tmp_G_24*tmp_G_8;
let tmp_G_26: f32 = tmp_G_19 + tmp_G_20;
let tmp_G_27: f32 = tmp_G_15 + tmp_G_23 - 1.0_f32;
let tmp_G_28: f32 = tmp_G_3 - tmp_G_4;
let tmp_G_29: f32 = 2.0_f32*tmp_G_14*tmp_G_28 - 2.0_f32*tmp_G_2*tmp_G_27 - 4.0_f32*tmp_G_26*tmp_G_8;
let tmp_G_30: f32 = q0*q1;
let tmp_G_31: f32 = q2*q3;
let tmp_G_32: f32 = tmp_G_30 - tmp_G_31;
let tmp_G_33: f32 = tmp_G_6 + tmp_G_7;
let tmp_G_34: f32 = 2.0_f32*q1*q1;
let tmp_G_35: f32 = tmp_G_13 + tmp_G_34;
let tmp_G_36: f32 = 2.0_f32*tmp_G_11*tmp_G_35 - 2.0_f32*tmp_G_17*tmp_G_33 - 4.0_f32*tmp_G_32*tmp_G_5;
let tmp_G_37: f32 = 4.0_f32*tmp_G_21*tmp_G_32 + 4.0_f32*tmp_G_22*tmp_G_33 + tmp_G_24*tmp_G_35;
let tmp_G_38: f32 = -2.0_f32*tmp_G_26*tmp_G_35 + 2.0_f32*tmp_G_27*tmp_G_32 - 4.0_f32*tmp_G_28*tmp_G_33;
let tmp_G_39: f32 = tmp_G_30 + tmp_G_31;
let tmp_G_40: f32 = tmp_G_12 + tmp_G_34 - 1.0_f32;
let tmp_G_41: f32 = tmp_G_0 - tmp_G_1;
let tmp_G_42: f32 = -4.0_f32*tmp_G_11*tmp_G_39 + 2.0_f32*tmp_G_17*tmp_G_41 - 2.0_f32*tmp_G_40*tmp_G_5;
let tmp_G_43: f32 = 2.0_f32*tmp_G_21*tmp_G_40 - 4.0_f32*tmp_G_22*tmp_G_41 - 2.0_f32*tmp_G_24*tmp_G_39;
let tmp_G_44: f32 = 4.0_f32*tmp_G_26*tmp_G_39 + tmp_G_27*tmp_G_40 + 4.0_f32*tmp_G_28*tmp_G_41;


// G
G[3][0] = tmp_G_18;
G[4][0] = tmp_G_36;
G[5][0] = tmp_G_42;
G[3][1] = tmp_G_25;
G[4][1] = tmp_G_37;
G[5][1] = tmp_G_43;
G[3][2] = tmp_G_29;
G[4][2] = tmp_G_38;
G[5][2] = tmp_G_44;
G[6][3] = tmp_G_18;
G[7][3] = tmp_G_36;
G[8][3] = tmp_G_42;
G[6][4] = tmp_G_25;
G[7][4] = tmp_G_37;
G[8][4] = tmp_G_43;
G[6][5] = tmp_G_29;
G[7][5] = tmp_G_38;
G[8][5] = tmp_G_44;
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
