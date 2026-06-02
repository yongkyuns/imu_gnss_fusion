// Generated first-order EKF attitude reset Jacobian block
const float tmp_reset0 = (1.0F/2.0F)*dtheta_z;
const float tmp_reset1 = (1.0F/2.0F)*dtheta_y;
const float tmp_reset2 = (1.0F/2.0F)*dtheta_x;


G_reset_theta[0][0] = 1;
G_reset_theta[1][0] = -tmp_reset0;
G_reset_theta[2][0] = tmp_reset1;
G_reset_theta[0][1] = tmp_reset0;
G_reset_theta[1][1] = 1;
G_reset_theta[2][1] = -tmp_reset2;
G_reset_theta[0][2] = -tmp_reset1;
G_reset_theta[1][2] = tmp_reset2;
G_reset_theta[2][2] = 1;
