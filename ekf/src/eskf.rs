use libm::sqrtf;

pub const NOMINAL_STATE_DIM: usize = 16;
pub const ERROR_STATE_DIM: usize = 15;

pub const IDX_DTHETA_X: usize = 0;
pub const IDX_DTHETA_Y: usize = 1;
pub const IDX_DTHETA_Z: usize = 2;
pub const IDX_DVEL_N: usize = 3;
pub const IDX_DVEL_E: usize = 4;
pub const IDX_DVEL_D: usize = 5;
pub const IDX_DPOS_N: usize = 6;
pub const IDX_DPOS_E: usize = 7;
pub const IDX_DPOS_D: usize = 8;
pub const IDX_DBG_X: usize = 9;
pub const IDX_DBG_Y: usize = 10;
pub const IDX_DBG_Z: usize = 11;
pub const IDX_DBA_X: usize = 12;
pub const IDX_DBA_Y: usize = 13;
pub const IDX_DBA_Z: usize = 14;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NominalState {
    pub q_bn: [f32; 4],
    pub vel_n: [f32; 3],
    pub pos_n: [f32; 3],
    pub gyro_bias_b: [f32; 3],
    pub accel_bias_b: [f32; 3],
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ErrorState {
    pub dtheta_b: [f32; 3],
    pub dvel_n: [f32; 3],
    pub dpos_n: [f32; 3],
    pub dgyro_bias_b: [f32; 3],
    pub daccel_bias_b: [f32; 3],
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImuDelta {
    pub dtheta_b: [f32; 3],
    pub dvel_b: [f32; 3],
    pub dt: f32,
}

impl NominalState {
    pub fn identity() -> Self {
        Self {
            q_bn: [1.0, 0.0, 0.0, 0.0],
            ..Self::default()
        }
    }

    pub fn inject_error(&mut self, dx: ErrorState) {
        let dq = quat_from_delta_theta(dx.dtheta_b);
        self.q_bn = quat_multiply(self.q_bn, dq);
        normalize_quat(&mut self.q_bn);
        add_assign3(&mut self.vel_n, dx.dvel_n);
        add_assign3(&mut self.pos_n, dx.dpos_n);
        add_assign3(&mut self.gyro_bias_b, dx.dgyro_bias_b);
        add_assign3(&mut self.accel_bias_b, dx.daccel_bias_b);
    }

    pub fn predict(&mut self, imu: ImuDelta, gravity_n: [f32; 3]) {
        let unbiased_dtheta = [
            imu.dtheta_b[0] - self.gyro_bias_b[0] * imu.dt,
            imu.dtheta_b[1] - self.gyro_bias_b[1] * imu.dt,
            imu.dtheta_b[2] - self.gyro_bias_b[2] * imu.dt,
        ];
        let unbiased_dvel_b = [
            imu.dvel_b[0] - self.accel_bias_b[0] * imu.dt,
            imu.dvel_b[1] - self.accel_bias_b[1] * imu.dt,
            imu.dvel_b[2] - self.accel_bias_b[2] * imu.dt,
        ];

        let dq = quat_from_delta_theta(unbiased_dtheta);
        self.q_bn = quat_multiply(self.q_bn, dq);
        normalize_quat(&mut self.q_bn);

        let dvel_n = mat3_mul_vec3(quat_to_rot(self.q_bn), unbiased_dvel_b);
        self.vel_n[0] += dvel_n[0] + gravity_n[0] * imu.dt;
        self.vel_n[1] += dvel_n[1] + gravity_n[1] * imu.dt;
        self.vel_n[2] += dvel_n[2] + gravity_n[2] * imu.dt;

        self.pos_n[0] += self.vel_n[0] * imu.dt;
        self.pos_n[1] += self.vel_n[1] * imu.dt;
        self.pos_n[2] += self.vel_n[2] * imu.dt;
    }
}

impl ErrorState {
    pub fn zeros() -> Self {
        Self::default()
    }

    pub fn from_array(dx: [f32; ERROR_STATE_DIM]) -> Self {
        Self {
            dtheta_b: [dx[IDX_DTHETA_X], dx[IDX_DTHETA_Y], dx[IDX_DTHETA_Z]],
            dvel_n: [dx[IDX_DVEL_N], dx[IDX_DVEL_E], dx[IDX_DVEL_D]],
            dpos_n: [dx[IDX_DPOS_N], dx[IDX_DPOS_E], dx[IDX_DPOS_D]],
            dgyro_bias_b: [dx[IDX_DBG_X], dx[IDX_DBG_Y], dx[IDX_DBG_Z]],
            daccel_bias_b: [dx[IDX_DBA_X], dx[IDX_DBA_Y], dx[IDX_DBA_Z]],
        }
    }

    pub fn to_array(self) -> [f32; ERROR_STATE_DIM] {
        [
            self.dtheta_b[0],
            self.dtheta_b[1],
            self.dtheta_b[2],
            self.dvel_n[0],
            self.dvel_n[1],
            self.dvel_n[2],
            self.dpos_n[0],
            self.dpos_n[1],
            self.dpos_n[2],
            self.dgyro_bias_b[0],
            self.dgyro_bias_b[1],
            self.dgyro_bias_b[2],
            self.daccel_bias_b[0],
            self.daccel_bias_b[1],
            self.daccel_bias_b[2],
        ]
    }
}

pub fn quat_from_delta_theta(dtheta_b: [f32; 3]) -> [f32; 4] {
    let dq = [
        1.0,
        0.5 * dtheta_b[0],
        0.5 * dtheta_b[1],
        0.5 * dtheta_b[2],
    ];
    normalize_quat_copy(dq)
}

pub fn quat_multiply(p: [f32; 4], q: [f32; 4]) -> [f32; 4] {
    [
        p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
        p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
        p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
        p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
    ]
}

pub fn normalize_quat(q: &mut [f32; 4]) {
    *q = normalize_quat_copy(*q);
}

pub fn quat_to_rot(q: [f32; 4]) -> [[f32; 3]; 3] {
    let q0 = q[0];
    let q1 = q[1];
    let q2 = q[2];
    let q3 = q[3];
    [
        [
            1.0 - 2.0 * q2 * q2 - 2.0 * q3 * q3,
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),
        ],
        [
            2.0 * (q1 * q2 + q0 * q3),
            1.0 - 2.0 * q1 * q1 - 2.0 * q3 * q3,
            2.0 * (q2 * q3 - q0 * q1),
        ],
        [
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            1.0 - 2.0 * q1 * q1 - 2.0 * q2 * q2,
        ],
    ]
}

pub fn error_reset_jacobian(dtheta_b: [f32; 3]) -> [[f32; ERROR_STATE_DIM]; ERROR_STATE_DIM] {
    let mut g = [[0.0; ERROR_STATE_DIM]; ERROR_STATE_DIM];
    let mut i = 0;
    while i < ERROR_STATE_DIM {
        g[i][i] = 1.0;
        i += 1;
    }

    let s = skew3(dtheta_b);
    let mut r = 0;
    while r < 3 {
        let mut c = 0;
        while c < 3 {
            g[r][c] -= 0.5 * s[r][c];
            c += 1;
        }
        r += 1;
    }
    g
}

fn normalize_quat_copy(q: [f32; 4]) -> [f32; 4] {
    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if n2 <= 1.0e-12 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        let inv_n = 1.0 / sqrtf(n2);
        [q[0] * inv_n, q[1] * inv_n, q[2] * inv_n, q[3] * inv_n]
    }
}

fn add_assign3(dst: &mut [f32; 3], src: [f32; 3]) {
    dst[0] += src[0];
    dst[1] += src[1];
    dst[2] += src[2];
}

fn mat3_mul_vec3(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn skew3(v: [f32; 3]) -> [[f32; 3]; 3] {
    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}
