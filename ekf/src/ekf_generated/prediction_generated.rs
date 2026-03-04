{
// Equations for state prediction.
let s_0 = 0.5*dax - 0.5*dax_b;
let s_1 = day - day_b;
let s_2 = 0.5*q2;
let s_3 = daz - daz_b;
let s_4 = 0.5*q3;
let s_5 = 0.5*q0;
let s_6 = 0.5*q1;
let s_7 = q0*q3;
let s_8 = q1*q2;
let s_9 = q0*q1;
let s_10 = q2*q3;
let s_11 = s_10 + s_9;
let s_12 = dt*g;
let s_13 = 2.0*s_12;
let s_14 = dvy - dvy_b + s_11*s_13;
let s_15 = 2.0*s_14;
let s_16 = q0*q2;
let s_17 = q1*q3;
let s_18 = 2.0*q1*q1;
let s_19 = 2.0*q2*q2 - 1.0;
let s_20 = s_18 + s_19;
let s_21 = -dvz + dvz_b + s_12*s_20;
let s_22 = 2.0*s_21;
let s_23 = 2.0*q3*q3;
let s_24 = s_16 - s_17;
let s_25 = -dvx + dvx_b + s_13*s_24;
let s_26 = 2.0*s_25;


ekf.state.q0 = q0 - q1*s_0 - s_1*s_2 - s_3*s_4;
ekf.state.q1 = q0*s_0 + q1 - s_1*s_4 + s_2*s_3;
ekf.state.q2 = q2 + q3*s_0 + s_1*s_5 - s_3*s_6;
ekf.state.q3 = -q2*s_0 + q3 + s_1*s_6 + s_3*s_5;
ekf.state.vn = -s_15*(s_7 - s_8) - s_22*(s_16 + s_17) + s_25*(s_19 + s_23) + vn;
ekf.state.ve = -s_14*(s_18 + s_23 - 1.0) + s_22*(-s_10 + s_9) - s_26*(s_7 + s_8) + ve;
ekf.state.vd = s_11*s_15 + s_20*s_21 + s_24*s_26 + vd;
ekf.state.pn = dt*vn + pn;
ekf.state.pe = dt*ve + pe;
ekf.state.pd = dt*vd + pd;
}
