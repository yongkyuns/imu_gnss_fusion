pub(crate) fn high_pass(
    current: f32,
    last_input: f32,
    last_output: f32,
    cutoff_hz: f32,
    dt: f32,
) -> f32 {
    let rc = 1.0 / (core::f32::consts::TAU * cutoff_hz.max(1.0e-3));
    let alpha = rc / (rc + dt);
    alpha * (last_output + current - last_input)
}

pub(crate) fn elapsed_since_last(last_t_s: &mut Option<f32>, t_s: f32) -> f32 {
    let dt = last_t_s
        .map(|last_t_s| (t_s - last_t_s).clamp(0.0, 0.2))
        .unwrap_or(0.0);
    *last_t_s = Some(t_s);
    dt
}

pub(crate) fn update_ema(previous: f32, value: f32, tau_s: f32, dt: f32) -> f32 {
    let alpha = dt / (tau_s.max(dt) + dt);
    (1.0 - alpha) * previous + alpha * value
}

pub(crate) fn update_abs_ema(previous: f32, value: f32, tau_s: f32, dt: f32) -> f32 {
    update_ema(previous, value.abs(), tau_s, dt)
}
