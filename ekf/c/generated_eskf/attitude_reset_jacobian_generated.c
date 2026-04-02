// Generated first-order ESKF attitude reset Jacobian block
const float ESKF_RESET0 = (1.0F/2.0F)*dtheta_z;
const float ESKF_RESET1 = (1.0F/2.0F)*dtheta_y;
const float ESKF_RESET2 = (1.0F/2.0F)*dtheta_x;


G_reset_theta[0][0] = 1;
G_reset_theta[1][0] = -ESKF_RESET0;
G_reset_theta[2][0] = ESKF_RESET1;
G_reset_theta[0][1] = ESKF_RESET0;
G_reset_theta[1][1] = 1;
G_reset_theta[2][1] = -ESKF_RESET2;
G_reset_theta[0][2] = -ESKF_RESET1;
G_reset_theta[1][2] = ESKF_RESET2;
G_reset_theta[2][2] = 1;


