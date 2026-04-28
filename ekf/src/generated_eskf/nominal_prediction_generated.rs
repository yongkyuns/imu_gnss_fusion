{
// Generated ESKF nominal-state prediction
let ESKF_PRED0: f32 = bgx*dt - dax;
let ESKF_PRED1: f32 = 0.5_f32*ESKF_PRED0;
let ESKF_PRED2: f32 = bgy*dt - day;
let ESKF_PRED3: f32 = 0.5_f32*ESKF_PRED2;
let ESKF_PRED4: f32 = bgz*dt - daz;
let ESKF_PRED5: f32 = 0.5_f32*ESKF_PRED4;
let ESKF_PRED6: f32 = q0*q3;
let ESKF_PRED7: f32 = q1*q2;
let ESKF_PRED8: f32 = q0*q1;
let ESKF_PRED9: f32 = q2*q3;
let ESKF_PRED10: f32 = ESKF_PRED8 + ESKF_PRED9;
let ESKF_PRED11: f32 = dt*g;
let ESKF_PRED12: f32 = 2.0_f32*ESKF_PRED11;
let ESKF_PRED13: f32 = ESKF_PRED10*ESKF_PRED12 - bay*dt + dvy;
let ESKF_PRED14: f32 = 2.0_f32*ESKF_PRED13;
let ESKF_PRED15: f32 = q0*q2;
let ESKF_PRED16: f32 = q1*q3;
let ESKF_PRED17: f32 = 2.0_f32*q1*q1;
let ESKF_PRED18: f32 = 2.0_f32*q2*q2 - 1.0_f32;
let ESKF_PRED19: f32 = ESKF_PRED17 + ESKF_PRED18;
let ESKF_PRED20: f32 = ESKF_PRED11*ESKF_PRED19 + baz*dt - dvz;
let ESKF_PRED21: f32 = 2.0_f32*ESKF_PRED20;
let ESKF_PRED22: f32 = 2.0_f32*q3*q3;
let ESKF_PRED23: f32 = ESKF_PRED15 - ESKF_PRED16;
let ESKF_PRED24: f32 = ESKF_PRED12*ESKF_PRED23 + bax*dt - dvx;
let ESKF_PRED25: f32 = 2.0_f32*ESKF_PRED24;


nominal.q0 = ESKF_PRED1*q1 + ESKF_PRED3*q2 + ESKF_PRED5*q3 + q0;
nominal.q1 = -ESKF_PRED1*q0 + 0.5_f32*ESKF_PRED2*q3 - ESKF_PRED5*q2 + q1;
nominal.q2 = -ESKF_PRED1*q3 - ESKF_PRED3*q0 + 0.5_f32*ESKF_PRED4*q1 + q2;
nominal.q3 = 0.5_f32*ESKF_PRED0*q2 - ESKF_PRED3*q1 - ESKF_PRED5*q0 + q3;
nominal.vn = -ESKF_PRED14*(ESKF_PRED6 - ESKF_PRED7) - ESKF_PRED21*(ESKF_PRED15 + ESKF_PRED16) + ESKF_PRED24*(ESKF_PRED18 + ESKF_PRED22) + vn;
nominal.ve = -ESKF_PRED13*(ESKF_PRED17 + ESKF_PRED22 - 1.0_f32) + ESKF_PRED21*(ESKF_PRED8 - ESKF_PRED9) - ESKF_PRED25*(ESKF_PRED6 + ESKF_PRED7) + ve;
nominal.vd = ESKF_PRED10*ESKF_PRED14 + ESKF_PRED19*ESKF_PRED20 + ESKF_PRED23*ESKF_PRED25 + vd;
nominal.pn = dt*vn + pn;
nominal.pe = dt*ve + pe;
nominal.pd = dt*vd + pd;
}
