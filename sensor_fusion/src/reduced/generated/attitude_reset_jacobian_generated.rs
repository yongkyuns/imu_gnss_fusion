{
// Generated first-order Reduced attitude reset Jacobian block
let REDUCED_RESET0: f32 = (1.0_f32/2.0_f32)*dtheta_z;
let REDUCED_RESET1: f32 = (1.0_f32/2.0_f32)*dtheta_y;
let REDUCED_RESET2: f32 = (1.0_f32/2.0_f32)*dtheta_x;


G_reset_theta[0][0] = 1.0_f32;
G_reset_theta[1][0] = -REDUCED_RESET0;
G_reset_theta[2][0] = REDUCED_RESET1;
G_reset_theta[0][1] = REDUCED_RESET0;
G_reset_theta[1][1] = 1.0_f32;
G_reset_theta[2][1] = -REDUCED_RESET2;
G_reset_theta[0][2] = -REDUCED_RESET1;
G_reset_theta[1][2] = REDUCED_RESET2;
G_reset_theta[2][2] = 1.0_f32;


}
