{
// Generated first-order ESKF attitude reset Jacobian block
let ESKF_RESET0: f32 = (1.0_f32/2.0_f32)*dtheta_z;
let ESKF_RESET1: f32 = (1.0_f32/2.0_f32)*dtheta_y;
let ESKF_RESET2: f32 = (1.0_f32/2.0_f32)*dtheta_x;


G_reset_theta[0][0] = 1.0_f32;
G_reset_theta[1][0] = -ESKF_RESET0;
G_reset_theta[2][0] = ESKF_RESET1;
G_reset_theta[0][1] = ESKF_RESET0;
G_reset_theta[1][1] = 1.0_f32;
G_reset_theta[2][1] = -ESKF_RESET2;
G_reset_theta[0][2] = -ESKF_RESET1;
G_reset_theta[1][2] = ESKF_RESET2;
G_reset_theta[2][2] = 1.0_f32;


}
