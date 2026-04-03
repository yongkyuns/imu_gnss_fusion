// Generated first-order loose INS/GNSS attitude reset Jacobian block
const float LOOSE_RESET0 = (1.0F/2.0F)*dtheta_z;
const float LOOSE_RESET1 = (1.0F/2.0F)*dtheta_y;
const float LOOSE_RESET2 = (1.0F/2.0F)*dtheta_x;


G_reset_theta[0][0] = 1;
G_reset_theta[1][0] = -LOOSE_RESET0;
G_reset_theta[2][0] = LOOSE_RESET1;
G_reset_theta[0][1] = LOOSE_RESET0;
G_reset_theta[1][1] = 1;
G_reset_theta[2][1] = -LOOSE_RESET2;
G_reset_theta[0][2] = -LOOSE_RESET1;
G_reset_theta[1][2] = LOOSE_RESET2;
G_reset_theta[2][2] = 1;


