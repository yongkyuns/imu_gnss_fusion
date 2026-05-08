{
// Generated first-order Full quaternion reset Jacobian block
let tmp_reset_0: f32 = (1.0_f32/2.0_f32)*dtheta_z;
let tmp_reset_1: f32 = (1.0_f32/2.0_f32)*dtheta_y;
let tmp_reset_2: f32 = (1.0_f32/2.0_f32)*dtheta_x;


G_reset_theta[0][0] = 1.0_f32;
G_reset_theta[1][0] = -tmp_reset_0;
G_reset_theta[2][0] = tmp_reset_1;
G_reset_theta[0][1] = tmp_reset_0;
G_reset_theta[1][1] = 1.0_f32;
G_reset_theta[2][1] = -tmp_reset_2;
G_reset_theta[0][2] = -tmp_reset_1;
G_reset_theta[1][2] = tmp_reset_2;
G_reset_theta[2][2] = 1.0_f32;


}
