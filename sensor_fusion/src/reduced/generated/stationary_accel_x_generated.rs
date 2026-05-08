{
// Sub Expressions
let tmp_hk_stat_ax0: f32 = q0*q0 - q1*q1 - q2*q2 + q3*q3;
let tmp_hk_stat_ax1: f32 = 1.0_f32*tmp_hk_stat_ax0;
let tmp_hk_stat_ax2: f32 = q0*q1 + q2*q3;
let tmp_hk_stat_ax3: f32 = 2.0_f32*tmp_hk_stat_ax2;
let tmp_hk_stat_ax4: f32 = g*g;
let tmp_hk_stat_ax5: f32 = -1.0_f32*tmp_hk_stat_ax0;
let tmp_hk_stat_ax6: f32 = R_STATIONARY_ACCEL - 1.0_f32*tmp_hk_stat_ax0*tmp_hk_stat_ax4*(P[1][1]*tmp_hk_stat_ax5 + P[1][2]*tmp_hk_stat_ax3) + tmp_hk_stat_ax3*tmp_hk_stat_ax4*(P[1][2]*tmp_hk_stat_ax5 + P[2][2]*tmp_hk_stat_ax3);
let tmp_hk_stat_ax7: f32 = -g/tmp_hk_stat_ax6;


// Observation Jacobians
H[1] = g*tmp_hk_stat_ax1;
H[2] = -g*tmp_hk_stat_ax3;


// Kalman gains
K[0] = tmp_hk_stat_ax7*(-P[0][1]*tmp_hk_stat_ax1 + 2.0_f32*P[0][2]*tmp_hk_stat_ax2);
K[1] = tmp_hk_stat_ax7*(-P[1][1]*tmp_hk_stat_ax1 + 2.0_f32*P[1][2]*tmp_hk_stat_ax2);
K[2] = tmp_hk_stat_ax7*(-P[1][2]*tmp_hk_stat_ax1 + 2.0_f32*P[2][2]*tmp_hk_stat_ax2);
K[3] = tmp_hk_stat_ax7*(-P[1][3]*tmp_hk_stat_ax1 + 2.0_f32*P[2][3]*tmp_hk_stat_ax2);
K[4] = tmp_hk_stat_ax7*(-P[1][4]*tmp_hk_stat_ax1 + 2.0_f32*P[2][4]*tmp_hk_stat_ax2);
K[5] = tmp_hk_stat_ax7*(-P[1][5]*tmp_hk_stat_ax1 + 2.0_f32*P[2][5]*tmp_hk_stat_ax2);
K[6] = tmp_hk_stat_ax7*(-P[1][6]*tmp_hk_stat_ax1 + 2.0_f32*P[2][6]*tmp_hk_stat_ax2);
K[7] = tmp_hk_stat_ax7*(-P[1][7]*tmp_hk_stat_ax1 + 2.0_f32*P[2][7]*tmp_hk_stat_ax2);
K[8] = tmp_hk_stat_ax7*(-P[1][8]*tmp_hk_stat_ax1 + 2.0_f32*P[2][8]*tmp_hk_stat_ax2);
K[9] = tmp_hk_stat_ax7*(-P[1][9]*tmp_hk_stat_ax1 + 2.0_f32*P[2][9]*tmp_hk_stat_ax2);
K[10] = tmp_hk_stat_ax7*(-P[1][10]*tmp_hk_stat_ax1 + 2.0_f32*P[2][10]*tmp_hk_stat_ax2);
K[11] = tmp_hk_stat_ax7*(-P[1][11]*tmp_hk_stat_ax1 + 2.0_f32*P[2][11]*tmp_hk_stat_ax2);
K[12] = tmp_hk_stat_ax7*(-P[1][12]*tmp_hk_stat_ax1 + 2.0_f32*P[2][12]*tmp_hk_stat_ax2);
K[13] = tmp_hk_stat_ax7*(-P[1][13]*tmp_hk_stat_ax1 + 2.0_f32*P[2][13]*tmp_hk_stat_ax2);
K[14] = tmp_hk_stat_ax7*(-P[1][14]*tmp_hk_stat_ax1 + 2.0_f32*P[2][14]*tmp_hk_stat_ax2);
K[15] = tmp_hk_stat_ax7*(-P[1][15]*tmp_hk_stat_ax1 + 2.0_f32*P[2][15]*tmp_hk_stat_ax2);
K[16] = tmp_hk_stat_ax7*(-P[1][16]*tmp_hk_stat_ax1 + 2.0_f32*P[2][16]*tmp_hk_stat_ax2);
K[17] = tmp_hk_stat_ax7*(-P[1][17]*tmp_hk_stat_ax1 + 2.0_f32*P[2][17]*tmp_hk_stat_ax2);


// Innovation Variance
S = tmp_hk_stat_ax6;
}
