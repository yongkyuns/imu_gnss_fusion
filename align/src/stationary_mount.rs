pub(crate) struct StationaryMountBootstrap {
    pub mean_accel_b: [f32; 3],
    pub c_b_v: [[f32; 3]; 3],
}

pub(crate) fn bootstrap_vehicle_to_body_from_stationary(
    accel_samples_b: &[[f32; 3]],
    yaw_seed_rad: f32,
) -> Result<StationaryMountBootstrap, &'static str> {
    bootstrap_vehicle_to_body_from_stationary_with_x_ref(
        accel_samples_b,
        yaw_seed_rad,
        [1.0, 0.0, 0.0],
    )
}

pub(crate) fn bootstrap_vehicle_to_body_from_stationary_with_x_ref(
    accel_samples_b: &[[f32; 3]],
    yaw_seed_rad: f32,
    mut x_ref: [f32; 3],
) -> Result<StationaryMountBootstrap, &'static str> {
    if accel_samples_b.is_empty() {
        return Err("stationary initialization requires samples");
    }

    let mut f_mean_b = [0.0_f32; 3];
    for sample in accel_samples_b {
        f_mean_b = vec3_add(f_mean_b, *sample);
    }
    f_mean_b = vec3_scale(f_mean_b, 1.0 / accel_samples_b.len() as f32);
    let n = vec3_norm(f_mean_b);
    if n < 1.0e-6 {
        return Err("stationary initialization requires nonzero accel mean");
    }

    let z_v_in_b = vec3_scale(f_mean_b, -1.0 / n);
    let mut x_v_in_b = vec3_sub(x_ref, vec3_scale(z_v_in_b, vec3_dot(z_v_in_b, x_ref)));
    if vec3_norm(x_v_in_b) < 1.0e-6 {
        x_ref = if x_ref[0].abs() > x_ref[1].abs() {
            [0.0, 1.0, 0.0]
        } else {
            [1.0, 0.0, 0.0]
        };
        x_v_in_b = vec3_sub(x_ref, vec3_scale(z_v_in_b, vec3_dot(z_v_in_b, x_ref)));
    }
    x_v_in_b = vec3_normalize(x_v_in_b).ok_or("failed to initialize x axis")?;
    let mut y_v_in_b = vec3_cross(z_v_in_b, x_v_in_b);
    y_v_in_b = vec3_normalize(y_v_in_b).ok_or("failed to initialize y axis")?;
    x_v_in_b = vec3_cross(y_v_in_b, z_v_in_b);
    x_v_in_b = vec3_normalize(x_v_in_b).ok_or("failed to orthonormalize x axis")?;

    let c_b_v_tilt = [
        [x_v_in_b[0], y_v_in_b[0], z_v_in_b[0]],
        [x_v_in_b[1], y_v_in_b[1], z_v_in_b[1]],
        [x_v_in_b[2], y_v_in_b[2], z_v_in_b[2]],
    ];
    let rpy = rot_to_euler_zyx(c_b_v_tilt);
    let dyaw = wrap_angle_rad(yaw_seed_rad - rpy[2]);
    let (s, c) = dyaw.sin_cos();
    let c_delta = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];

    Ok(StationaryMountBootstrap {
        mean_accel_b: f_mean_b,
        c_b_v: mat3_mul(c_b_v_tilt, c_delta),
    })
}

fn rot_to_euler_zyx(c: [[f32; 3]; 3]) -> [f32; 3] {
    let pitch = (-c[2][0]).asin();
    let roll = c[2][1].atan2(c[2][2]);
    let yaw = wrap_angle_rad(c[1][0].atan2(c[0][0]));
    [roll, pitch, yaw]
}

fn wrap_angle_rad(x: f32) -> f32 {
    x.sin().atan2(x.cos())
}

fn vec3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vec3_norm(v: [f32; 3]) -> f32 {
    vec3_dot(v, v).sqrt()
}

fn vec3_normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let n = vec3_norm(v);
    if n < 1.0e-9 {
        None
    } else {
        Some(vec3_scale(v, 1.0 / n))
    }
}

fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}
