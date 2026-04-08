use sensor_fusion::c_api::{CLooseImuDelta, CLooseWrapper};
use sensor_fusion::loose::LoosePredictNoise;

fn assert_close(a: f32, b: f32, tol: f32, ctx: &str) {
    let d = (a - b).abs();
    assert!(d <= tol, "{ctx}: |{a} - {b}| = {d} > {tol}");
}

fn assert_nominal_close(
    lhs: &sensor_fusion::c_api::CLooseNominalState,
    rhs: &sensor_fusion::c_api::CLooseNominalState,
    tol: f32,
    ctx: &str,
) {
    assert_close(lhs.q0, rhs.q0, tol, &format!("{ctx}.q0"));
    assert_close(lhs.q1, rhs.q1, tol, &format!("{ctx}.q1"));
    assert_close(lhs.q2, rhs.q2, tol, &format!("{ctx}.q2"));
    assert_close(lhs.q3, rhs.q3, tol, &format!("{ctx}.q3"));
    assert_close(lhs.pn, rhs.pn, tol, &format!("{ctx}.pn"));
    assert_close(lhs.pe, rhs.pe, tol, &format!("{ctx}.pe"));
    assert_close(lhs.pd, rhs.pd, tol, &format!("{ctx}.pd"));
    assert_close(lhs.vn, rhs.vn, tol, &format!("{ctx}.vn"));
    assert_close(lhs.ve, rhs.ve, tol, &format!("{ctx}.ve"));
    assert_close(lhs.vd, rhs.vd, tol, &format!("{ctx}.vd"));
    assert_close(lhs.bgx, rhs.bgx, tol, &format!("{ctx}.bgx"));
    assert_close(lhs.bgy, rhs.bgy, tol, &format!("{ctx}.bgy"));
    assert_close(lhs.bgz, rhs.bgz, tol, &format!("{ctx}.bgz"));
    assert_close(lhs.bax, rhs.bax, tol, &format!("{ctx}.bax"));
    assert_close(lhs.bay, rhs.bay, tol, &format!("{ctx}.bay"));
    assert_close(lhs.baz, rhs.baz, tol, &format!("{ctx}.baz"));
    assert_close(lhs.sgx, rhs.sgx, tol, &format!("{ctx}.sgx"));
    assert_close(lhs.sgy, rhs.sgy, tol, &format!("{ctx}.sgy"));
    assert_close(lhs.sgz, rhs.sgz, tol, &format!("{ctx}.sgz"));
    assert_close(lhs.sax, rhs.sax, tol, &format!("{ctx}.sax"));
    assert_close(lhs.say, rhs.say, tol, &format!("{ctx}.say"));
    assert_close(lhs.saz, rhs.saz, tol, &format!("{ctx}.saz"));
    assert_close(lhs.qcs0, rhs.qcs0, tol, &format!("{ctx}.qcs0"));
    assert_close(lhs.qcs1, rhs.qcs1, tol, &format!("{ctx}.qcs1"));
    assert_close(lhs.qcs2, rhs.qcs2, tol, &format!("{ctx}.qcs2"));
    assert_close(lhs.qcs3, rhs.qcs3, tol, &format!("{ctx}.qcs3"));
}

fn assert_covariance_close(
    lhs: &[[f32; 24]; 24],
    rhs: &[[f32; 24]; 24],
    tol: f32,
    ctx: &str,
) {
    for i in 0..24 {
        for j in 0..24 {
            assert_close(lhs[i][j], rhs[i][j], tol, &format!("{ctx}.p[{i}][{j}]"));
        }
    }
}

const WGS84_A: f32 = 6_378_137.0;
const WGS84_OMEGA_IE: f32 = 7.292_115e-5;
const WGS84_GM: f32 = 3.986_004_4e14;
const WGS84_J2: f32 = 1.082_629_8e-3;

#[derive(Clone, Copy)]
struct RefLooseState {
    q_es: [f32; 4],
    pos_e: [f32; 3],
    vel_e: [f32; 3],
    b_w: [f32; 3],
    b_f: [f32; 3],
    s_w: [f32; 3],
    s_f: [f32; 3],
}

#[derive(Clone, Copy)]
struct RefLooseFullState {
    q_es: [f32; 4],
    pos_e: [f32; 3],
    vel_e: [f32; 3],
    b_f: [f32; 3],
    b_w: [f32; 3],
    s_f: [f32; 3],
    s_w: [f32; 3],
    q_cs: [f32; 4],
}

fn quat_mul(p: [f32; 4], q: [f32; 4]) -> [f32; 4] {
    [
        p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
        p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
        p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
        p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
    ]
}

fn quat_normalize(mut q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n <= 1.0e-12 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    q[0] /= n;
    q[1] /= n;
    q[2] /= n;
    q[3] /= n;
    q
}

fn quat_to_dcm(q: [f32; 4]) -> [[f32; 3]; 3] {
    let q = quat_normalize(q);
    let q1_2 = q[1] * q[1];
    let q2_2 = q[2] * q[2];
    let q3_2 = q[3] * q[3];
    [
        [
            1.0 - 2.0 * (q2_2 + q3_2),
            2.0 * (q[1] * q[2] - q[0] * q[3]),
            2.0 * (q[1] * q[3] + q[0] * q[2]),
        ],
        [
            2.0 * (q[1] * q[2] + q[0] * q[3]),
            1.0 - 2.0 * (q1_2 + q3_2),
            2.0 * (q[2] * q[3] - q[0] * q[1]),
        ],
        [
            2.0 * (q[1] * q[3] - q[0] * q[2]),
            2.0 * (q[2] * q[3] + q[0] * q[1]),
            1.0 - 2.0 * (q1_2 + q2_2),
        ],
    ]
}

fn skew(v: [f32; 3]) -> [[f32; 3]; 3] {
    [
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ]
}

fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

fn mat3_transpose(a: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

fn mat3_vec(a: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

fn ecef_to_llh(x_ecef: [f32; 3]) -> [f32; 3] {
    let a2 = WGS84_A * WGS84_A;
    let wgs84_b = 6_356_752.314_245_f32;
    let b2 = wgs84_b * wgs84_b;
    let wgs84_e2 = 1.0 - b2 / a2;
    let z2 = x_ecef[2] * x_ecef[2];
    let r2 = x_ecef[0] * x_ecef[0] + x_ecef[1] * x_ecef[1];
    let r = r2.sqrt();
    let f = 54.0 * b2 * z2;
    let g = r2 + (1.0 - wgs84_e2) * z2 - wgs84_e2 * (a2 - b2);
    let tmp = wgs84_e2 * wgs84_e2;
    let c = tmp * f * r2 / (g * g * g);
    let s = (1.0 + c + (c * c + 2.0 * c).sqrt()).powf(1.0 / 3.0);
    let p = f / (3.0 * (s + 1.0 / s + 1.0).powi(2) * g * g);
    let q = (1.0 + 2.0 * tmp * p).sqrt();
    let r0 = -p * wgs84_e2 * r / (1.0 + q)
        + (0.5 * a2 * (1.0 + 1.0 / q)
            - p * (1.0 - wgs84_e2) * z2 / (q * (1.0 + q))
            - 0.5 * p * r2)
            .sqrt();
    let tmp2 = (r - wgs84_e2 * r0).powi(2);
    let u = (tmp2 + z2).sqrt();
    let v = (tmp2 + (1.0 - wgs84_e2) * z2).sqrt();
    let tmp3 = 1.0 / (WGS84_A * v);
    let z0 = b2 * x_ecef[2] * tmp3;
    let height = u * (1.0 - b2 * tmp3);
    let lat = (x_ecef[2] + (a2 / b2 - 1.0) * z0).atan2(r);
    let lon = x_ecef[1].atan2(x_ecef[0]);
    [lat, lon, height]
}

fn dcm_ecef_to_ned(lat: f32, lon: f32) -> [[f32; 3]; 3] {
    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let sin_lon = lon.sin();
    let cos_lon = lon.cos();
    [
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [-sin_lon, cos_lon, 0.0],
        [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat],
    ]
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn chi2_scalar(res: f32, p: &[[f32; 24]; 24], h: &[f32; 24], r: f32) -> bool {
    let mut s = r;
    for i in 0..24 {
        for j in 0..24 {
            s += h[i] * p[i][j] * h[j];
        }
    }
    res.abs() > 3.0 * s.sqrt()
}

fn chi2_vec3(res: [f32; 3], p: &[[f32; 24]; 24], h: &[[f32; 24]; 3], r: [[f32; 3]; 3]) -> bool {
    for idx in 0..3 {
        let mut s = r[idx][idx];
        for i in 0..24 {
            for j in 0..24 {
                s += h[idx][i] * p[i][j] * h[idx][j];
            }
        }
        if res[idx].abs() > 3.0 * s.sqrt() {
            return true;
        }
    }
    false
}

fn cholesky_lower_3x3(a: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut l = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                l[i][j] = sum.max(1.0e-9).sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    l
}

fn inv_upper_from_lower_transpose(l: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut t = [[0.0; 3]; 3];
    for col in 0..3 {
        for i in (0..3).rev() {
            let rhs = if i == col { 1.0 } else { 0.0 };
            let mut accum = rhs;
            for k in i + 1..3 {
                accum -= l[k][i] * t[k][col];
            }
            t[i][col] = accum / l[i][i];
        }
    }
    t
}

fn inject_ref_error(mut x: RefLooseFullState, dx: [f32; 24]) -> RefLooseFullState {
    x.pos_e[0] += dx[0];
    x.pos_e[1] += dx[1];
    x.pos_e[2] += dx[2];
    x.vel_e[0] += dx[3];
    x.vel_e[1] += dx[4];
    x.vel_e[2] += dx[5];
    x.b_f[0] += dx[9];
    x.b_f[1] += dx[10];
    x.b_f[2] += dx[11];
    x.b_w[0] += dx[12];
    x.b_w[1] += dx[13];
    x.b_w[2] += dx[14];
    x.s_f[0] += dx[15];
    x.s_f[1] += dx[16];
    x.s_f[2] += dx[17];
    x.s_w[0] += dx[18];
    x.s_w[1] += dx[19];
    x.s_w[2] += dx[20];
    let dq_es = quat_normalize([1.0, 0.5 * dx[6], 0.5 * dx[7], 0.5 * dx[8]]);
    let dq_cs = quat_normalize([1.0, 0.5 * dx[21], 0.5 * dx[22], 0.5 * dx[23]]);
    x.q_es = quat_normalize(quat_mul(dq_es, x.q_es));
    x.q_cs = quat_normalize(quat_mul(dq_cs, x.q_cs));
    x
}

fn reference_batch_update(
    x: RefLooseFullState,
    p_in: [[f32; 24]; 24],
    pos_ecef_m: Option<[f32; 3]>,
    h_acc_m: f32,
    dt_since_last_gnss_s: f32,
    gyro_radps: [f32; 3],
    accel_mps2: [f32; 3],
    dt_s: f32,
) -> (RefLooseFullState, [[f32; 24]; 24]) {
    const REFERENCE_GYRO_DT_S: f32 = 0.02;
    let mut h_rows = [[0.0_f32; 24]; 5];
    let mut residuals = [0.0_f32; 5];
    let mut variances = [0.0_f32; 5];
    let mut obs_count = 0usize;

    if let Some(pos_ecef_m) = pos_ecef_m {
        let llh = ecef_to_llh(x.pos_e);
        let c_en = dcm_ecef_to_ned(llh[0], llh[1]);
        let r_n = [
            [h_acc_m * h_acc_m, 0.0, 0.0],
            [0.0, h_acc_m * h_acc_m, 0.0],
            [0.0, 0.0, (2.5 * h_acc_m) * (2.5 * h_acc_m)],
        ];
        let c_ne = mat3_transpose(c_en);
        let r_e = mat3_mul(mat3_mul(c_ne, r_n), c_en);
        let l = cholesky_lower_3x3(r_e);
        let t = inv_upper_from_lower_transpose(l);
        let x_meas = mat3_vec(t, pos_ecef_m);
        let x_est = mat3_vec(t, x.pos_e);
        let mut h_tmp = [[0.0_f32; 24]; 3];
        for i in 0..3 {
            for j in 0..3 {
                h_tmp[i][j] = t[i][j];
            }
        }
        let residual = [
            x_meas[0] - x_est[0],
            x_meas[1] - x_est[1],
            x_meas[2] - x_est[2],
        ];
        if !chi2_vec3(residual, &p_in, &h_tmp, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) {
            let d_ttag_s = if dt_since_last_gnss_s == 0.0 || dt_since_last_gnss_s >= 1.0 {
                1.0
            } else {
                dt_since_last_gnss_s
            };
            for row in 0..3 {
                h_rows[obs_count] = h_tmp[row];
                residuals[obs_count] = residual[row];
                variances[obs_count] = 1.0 / d_ttag_s;
                obs_count += 1;
            }
        }
    }

    let omega_is = [
        x.s_w[0] * gyro_radps[0] + x.b_w[0],
        x.s_w[1] * gyro_radps[1] + x.b_w[1],
        x.s_w[2] * gyro_radps[2] + x.b_w[2],
    ];
    let f_s = [
        x.s_f[0] * accel_mps2[0] + x.b_f[0],
        x.s_f[1] * accel_mps2[1] + x.b_f[1],
        x.s_f[2] * accel_mps2[2] + x.b_f[2],
    ];
    if norm3(omega_is) < 0.03 && (norm3(f_s) - 9.81).abs() < 0.2 {
        let c_ce = mat3_mul(quat_to_dcm(x.q_cs), mat3_transpose(quat_to_dcm(x.q_es)));
        let v_c = mat3_vec(c_ce, x.vel_e);
        let sv = skew(x.vel_e);
        let cv = mat3_mul(c_ce, sv);
        let sm = skew([-v_c[0], -v_c[1], -v_c[2]]);
        let mut h_tmp = [[0.0_f32; 24]; 3];
        for i in 0..3 {
            for j in 0..3 {
                h_tmp[i][3 + j] = c_ce[i][j];
                h_tmp[i][6 + j] = cv[i][j];
                h_tmp[i][21 + j] = sm[i][j];
            }
        }
        let _ = dt_s;
        let gate_var_y = 0.1 * 0.1;
        let gate_var_z = 0.05 * 0.05;
        let var_y = gate_var_y / REFERENCE_GYRO_DT_S;
        let var_z = gate_var_z / REFERENCE_GYRO_DT_S;
        if !chi2_scalar(-v_c[1], &p_in, &h_tmp[1], gate_var_y) {
            h_rows[obs_count] = h_tmp[1];
            residuals[obs_count] = -v_c[1];
            variances[obs_count] = var_y;
            obs_count += 1;
        }
        if !chi2_scalar(-v_c[2], &p_in, &h_tmp[2], gate_var_z) {
            h_rows[obs_count] = h_tmp[2];
            residuals[obs_count] = -v_c[2];
            variances[obs_count] = var_z;
            obs_count += 1;
        }
    }

    let mut p = p_in;
    let mut dx = [0.0_f32; 24];
    for obs in 0..obs_count {
        let mut k = [0.0_f32; 24];
        let mut s = variances[obs];
        for i in 0..24 {
            for j in 0..24 {
                k[i] += p[i][j] * h_rows[obs][j];
            }
        }
        for i in 0..24 {
            s += h_rows[obs][i] * k[i];
        }
        if s <= 0.0 {
            continue;
        }
        let mut hd = 0.0;
        for i in 0..24 {
            hd += h_rows[obs][i] * dx[i];
            k[i] /= s;
        }
        for i in 0..24 {
            dx[i] += k[i] * (residuals[obs] - hd);
        }
        let mut ikh = [[0.0_f32; 24]; 24];
        for i in 0..24 {
            for j in 0..24 {
                ikh[i][j] = if i == j { 1.0 } else { 0.0 } - k[i] * h_rows[obs][j];
            }
        }
        let mut p_new = [[0.0_f32; 24]; 24];
        for i in 0..24 {
            for j in 0..24 {
                let mut accum = 0.0;
                for a in 0..24 {
                    for b in 0..24 {
                        accum += ikh[i][a] * p[a][b] * ikh[j][b];
                    }
                }
                p_new[i][j] = accum + variances[obs] * k[i] * k[j];
            }
        }
        p = p_new;
    }
    (inject_ref_error(x, dx), p)
}

fn reference_transition(
    q_es: [f32; 4],
    b_f: [f32; 3],
    b_w: [f32; 3],
    s_f: [f32; 3],
    s_w: [f32; 3],
    accel_raw: [f32; 3],
    gyro_raw: [f32; 3],
    dt: f32,
) -> ([[f32; 24]; 24], [[f32; 21]; 24]) {
    let c_es = quat_to_dcm(q_es);
    let f_s = [
        s_f[0] * accel_raw[0] + b_f[0],
        s_f[1] * accel_raw[1] + b_f[1],
        s_f[2] * accel_raw[2] + b_f[2],
    ];
    let mut f = [[0.0_f32; 24]; 24];
    for (i, row) in f.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for i in 0..3 {
        f[i][3 + i] += dt;
    }
    let vv = skew([0.0, 0.0, -2.0 * WGS84_OMEGA_IE]);
    let aa = skew(mat3_vec(c_es, [-f_s[0], -f_s[1], -f_s[2]]));
    let ww = skew([0.0, 0.0, -WGS84_OMEGA_IE]);
    for i in 0..3 {
        for j in 0..3 {
            f[3 + i][3 + j] += dt * vv[i][j];
            f[3 + i][6 + j] += dt * aa[i][j];
            f[3 + i][9 + j] += dt * c_es[i][j];
            f[3 + i][15 + j] += dt * c_es[i][j] * accel_raw[j];
            f[6 + i][6 + j] += dt * ww[i][j];
            f[6 + i][12 + j] += dt * c_es[i][j];
            f[6 + i][18 + j] += dt * c_es[i][j] * gyro_raw[j];
        }
    }

    let mut g = [[0.0_f32; 21]; 24];
    for i in 0..3 {
        for j in 0..3 {
            g[3 + i][j] = c_es[i][j];
            g[6 + i][3 + j] = c_es[i][j];
        }
        g[9 + i][6 + i] = 1.0;
        g[12 + i][9 + i] = 1.0;
        g[15 + i][12 + i] = 1.0;
        g[18 + i][15 + i] = 1.0;
        g[21 + i][18 + i] = 1.0;
    }
    (f, g)
}

fn mat24_mul(a: &[[f32; 24]; 24], b: &[[f32; 24]; 24]) -> [[f32; 24]; 24] {
    let mut out = [[0.0_f32; 24]; 24];
    for i in 0..24 {
        for j in 0..24 {
            for k in 0..24 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

fn mat24_t(a: &[[f32; 24]; 24]) -> [[f32; 24]; 24] {
    let mut out = [[0.0_f32; 24]; 24];
    for i in 0..24 {
        for j in 0..24 {
            out[i][j] = a[j][i];
        }
    }
    out
}

fn mat24x21_diag_mat21x24(g: &[[f32; 21]; 24], q_diag: [f32; 21]) -> [[f32; 24]; 24] {
    let mut out = [[0.0_f32; 24]; 24];
    for i in 0..24 {
        for j in 0..24 {
            let mut accum = 0.0;
            for k in 0..21 {
                accum += g[i][k] * q_diag[k] * g[j][k];
            }
            out[i][j] = accum;
        }
    }
    out
}

fn gravity_ecef_j2(x_e: [f32; 3]) -> [f32; 3] {
    let r = (x_e[0] * x_e[0] + x_e[1] * x_e[1] + x_e[2] * x_e[2]).sqrt();
    if r <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let r2 = r * r;
    let r3 = r * r2;
    let tmp1 = WGS84_GM / r3;
    let tmp2 = 1.5 * (WGS84_A * (WGS84_A * WGS84_J2)) / r2;
    let tmp3 = 5.0 * x_e[2] * x_e[2] / r2;
    [
        tmp1 * (-x_e[0] - tmp2 * (x_e[0] - tmp3 * x_e[0])) + WGS84_OMEGA_IE * WGS84_OMEGA_IE * x_e[0],
        tmp1 * (-x_e[1] - tmp2 * (x_e[1] - tmp3 * x_e[1])) + WGS84_OMEGA_IE * WGS84_OMEGA_IE * x_e[1],
        tmp1 * (-x_e[2] - tmp2 * (3.0 * x_e[2] - tmp3 * x_e[2])),
    ]
}

fn ode(x: RefLooseState, omega_is: [f32; 3], f_s: [f32; 3]) -> ([f32; 3], [f32; 3], [f32; 4]) {
    let c_es = quat_to_dcm(x.q_es);
    let f_e = [
        c_es[0][0] * f_s[0] + c_es[0][1] * f_s[1] + c_es[0][2] * f_s[2],
        c_es[1][0] * f_s[0] + c_es[1][1] * f_s[1] + c_es[1][2] * f_s[2],
        c_es[2][0] * f_s[0] + c_es[2][1] * f_s[1] + c_es[2][2] * f_s[2],
    ];
    let g_e = gravity_ecef_j2(x.pos_e);
    let v_dot = [
        g_e[0] + f_e[0] + 2.0 * WGS84_OMEGA_IE * x.vel_e[1],
        g_e[1] + f_e[1] - 2.0 * WGS84_OMEGA_IE * x.vel_e[0],
        g_e[2] + f_e[2],
    ];
    let q_omega = [0.0, omega_is[0], omega_is[1], omega_is[2]];
    let q_mul = quat_mul(x.q_es, q_omega);
    let q_dot = [
        0.5 * (q_mul[0] + WGS84_OMEGA_IE * x.q_es[3]),
        0.5 * (q_mul[1] + WGS84_OMEGA_IE * x.q_es[2]),
        0.5 * (q_mul[2] - WGS84_OMEGA_IE * x.q_es[1]),
        0.5 * (q_mul[3] - WGS84_OMEGA_IE * x.q_es[0]),
    ];
    (x.vel_e, v_dot, q_dot)
}

fn ref_two_sample_heun_step(
    x: RefLooseState,
    omega_1_raw: [f32; 3],
    omega_2_raw: [f32; 3],
    f_1_raw: [f32; 3],
    f_2_raw: [f32; 3],
    dt: f32,
) -> RefLooseState {
    let omega_1 = [
        x.s_w[0] * omega_1_raw[0] + x.b_w[0],
        x.s_w[1] * omega_1_raw[1] + x.b_w[1],
        x.s_w[2] * omega_1_raw[2] + x.b_w[2],
    ];
    let omega_2 = [
        x.s_w[0] * omega_2_raw[0] + x.b_w[0],
        x.s_w[1] * omega_2_raw[1] + x.b_w[1],
        x.s_w[2] * omega_2_raw[2] + x.b_w[2],
    ];
    let f_1 = [
        x.s_f[0] * f_1_raw[0] + x.b_f[0],
        x.s_f[1] * f_1_raw[1] + x.b_f[1],
        x.s_f[2] * f_1_raw[2] + x.b_f[2],
    ];
    let f_2 = [
        x.s_f[0] * f_2_raw[0] + x.b_f[0],
        x.s_f[1] * f_2_raw[1] + x.b_f[1],
        x.s_f[2] * f_2_raw[2] + x.b_f[2],
    ];

    let (pos_dot_1, vel_dot_1, q_dot_1) = ode(x, omega_1, f_1);
    let x_tmp = RefLooseState {
        q_es: quat_normalize([
            x.q_es[0] + dt * q_dot_1[0],
            x.q_es[1] + dt * q_dot_1[1],
            x.q_es[2] + dt * q_dot_1[2],
            x.q_es[3] + dt * q_dot_1[3],
        ]),
        pos_e: [
            x.pos_e[0] + dt * pos_dot_1[0],
            x.pos_e[1] + dt * pos_dot_1[1],
            x.pos_e[2] + dt * pos_dot_1[2],
        ],
        vel_e: [
            x.vel_e[0] + dt * vel_dot_1[0],
            x.vel_e[1] + dt * vel_dot_1[1],
            x.vel_e[2] + dt * vel_dot_1[2],
        ],
        ..x
    };
    let (pos_dot_2, vel_dot_2, q_dot_2) = ode(x_tmp, omega_2, f_2);

    RefLooseState {
        q_es: quat_normalize([
            x.q_es[0] + 0.5 * dt * (q_dot_1[0] + q_dot_2[0]),
            x.q_es[1] + 0.5 * dt * (q_dot_1[1] + q_dot_2[1]),
            x.q_es[2] + 0.5 * dt * (q_dot_1[2] + q_dot_2[2]),
            x.q_es[3] + 0.5 * dt * (q_dot_1[3] + q_dot_2[3]),
        ]),
        pos_e: [
            x.pos_e[0] + 0.5 * dt * (pos_dot_1[0] + pos_dot_2[0]),
            x.pos_e[1] + 0.5 * dt * (pos_dot_1[1] + pos_dot_2[1]),
            x.pos_e[2] + 0.5 * dt * (pos_dot_1[2] + pos_dot_2[2]),
        ],
        vel_e: [
            x.vel_e[0] + 0.5 * dt * (vel_dot_1[0] + vel_dot_2[0]),
            x.vel_e[1] + 0.5 * dt * (vel_dot_1[1] + vel_dot_2[1]),
            x.vel_e[2] + 0.5 * dt * (vel_dot_1[2] + vel_dot_2[2]),
        ],
        ..x
    }
}

#[test]
fn loose_reference_ecef_init_copies_nominal_state_and_covariance_diagonal() {
    let noise = LoosePredictNoise::default();
    let mut loose = CLooseWrapper::new(noise);
    let p_diag = core::array::from_fn(|i| 0.01 * (i as f32 + 1.0));

    loose.init_from_reference_ecef_state(
        [0.7, 0.1, 0.2, 0.67],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [0.01, 0.02, 0.03],
        [0.04, 0.05, 0.06],
        [1.1, 1.2, 1.3],
        [0.9, 0.8, 0.7],
        [0.99, 0.01, 0.02, 0.03],
        Some(p_diag),
    );

    let x = loose.nominal();
    let p = loose.covariance();
    assert_close(x.q0, 0.7, 1.0e-6, "q0");
    assert_close(x.q3, 0.67, 1.0e-6, "q3");
    assert_close(x.pn, 1.0, 1.0e-6, "pn");
    assert_close(x.ve, 5.0, 1.0e-6, "ve");
    assert_close(x.bgz, 0.03, 1.0e-6, "bgz");
    assert_close(x.bay, 0.05, 1.0e-6, "bay");
    assert_close(x.sgx, 1.1, 1.0e-6, "sgx");
    assert_close(x.saz, 0.7, 1.0e-6, "saz");
    assert_close(x.qcs2, 0.02, 1.0e-6, "qcs2");
    assert_close(p[0][0], 0.01, 1.0e-6, "p00");
    assert_close(p[9][9], 0.10, 1.0e-6, "p99");
    assert_close(p[23][23], 0.24, 1.0e-6, "p2323");
}

#[test]
fn loose_predict_with_zero_dt_is_noop() {
    let noise = LoosePredictNoise::default();
    let mut loose = CLooseWrapper::new(noise);
    let p_diag = core::array::from_fn(|i| 0.02 * (i as f32 + 1.0));

    loose.init_from_reference_ecef_state(
        [1.0, 0.0, 0.0, 0.0],
        [6378137.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [0.01, 0.02, 0.03],
        [0.04, 0.05, 0.06],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        Some(p_diag),
    );
    let before = *loose.nominal();
    let p00_before = loose.covariance()[0][0];
    let p99_before = loose.covariance()[9][9];

    loose.predict(CLooseImuDelta {
        dax_1: 1.0,
        day_1: -2.0,
        daz_1: 3.0,
        dvx_1: 4.0,
        dvy_1: -5.0,
        dvz_1: 6.0,
        dax_2: 1.0,
        day_2: -2.0,
        daz_2: 3.0,
        dvx_2: 4.0,
        dvy_2: -5.0,
        dvz_2: 6.0,
        dt: 0.0,
    });

    let after = loose.nominal();
    assert_close(after.q0, before.q0, 1.0e-6, "q0");
    assert_close(after.q1, before.q1, 1.0e-6, "q1");
    assert_close(after.q2, before.q2, 1.0e-6, "q2");
    assert_close(after.q3, before.q3, 1.0e-6, "q3");
    assert_close(after.vn, before.vn, 1.0e-6, "vn");
    assert_close(after.pe, before.pe, 1.0e-6, "pe");
    assert_close(loose.covariance()[0][0], p00_before, 1.0e-6, "p00");
    assert_close(loose.covariance()[9][9], p99_before, 1.0e-6, "p99");
}

#[test]
fn loose_single_delta_predict_matches_reference_two_sample_heun_for_constant_imu() {
    let noise = LoosePredictNoise::default();
    let mut loose = CLooseWrapper::new(noise);
    let dt = 0.01;
    let state = RefLooseState {
        q_es: [1.0, 0.0, 0.0, 0.0],
        pos_e: [WGS84_A, 0.0, 0.0],
        vel_e: [1.0, -2.0, 0.5],
        b_w: [0.01, -0.02, 0.03],
        b_f: [0.1, -0.2, 0.3],
        s_w: [1.0, 1.0, 1.0],
        s_f: [1.0, 1.0, 1.0],
    };
    let omega = [0.2, -0.1, 0.05];
    let accel = [9.7, 0.2, -0.3];
    let expected = ref_two_sample_heun_step(state, omega, omega, accel, accel, dt);

    loose.init_from_reference_ecef_state(
        state.q_es,
        [state.pos_e[0] as f64, state.pos_e[1] as f64, state.pos_e[2] as f64],
        state.vel_e,
        state.b_w,
        state.b_f,
        state.s_w,
        state.s_f,
        [1.0, 0.0, 0.0, 0.0],
        None,
    );
    loose.predict(CLooseImuDelta {
        dax_1: omega[0] * dt,
        day_1: omega[1] * dt,
        daz_1: omega[2] * dt,
        dvx_1: accel[0] * dt,
        dvy_1: accel[1] * dt,
        dvz_1: accel[2] * dt,
        dax_2: omega[0] * dt,
        day_2: omega[1] * dt,
        daz_2: omega[2] * dt,
        dvx_2: accel[0] * dt,
        dvy_2: accel[1] * dt,
        dvz_2: accel[2] * dt,
        dt,
    });

    let x = loose.nominal();
    assert_close(x.q0, expected.q_es[0], 1.0e-5, "q0");
    assert_close(x.q1, expected.q_es[1], 1.0e-5, "q1");
    assert_close(x.q2, expected.q_es[2], 1.0e-5, "q2");
    assert_close(x.q3, expected.q_es[3], 1.0e-5, "q3");
    assert_close(x.pn, expected.pos_e[0], 1.0e-3, "pn");
    assert_close(x.pe, expected.pos_e[1], 1.0e-4, "pe");
    assert_close(x.pd, expected.pos_e[2], 1.0e-4, "pd");
    assert_close(x.vn, expected.vel_e[0], 1.0e-4, "vn");
    assert_close(x.ve, expected.vel_e[1], 1.0e-4, "ve");
    assert_close(x.vd, expected.vel_e[2], 1.0e-4, "vd");
}

#[test]
fn loose_two_sample_predict_matches_reference_when_imu_changes_within_step() {
    let noise = LoosePredictNoise::default();
    let mut loose = CLooseWrapper::new(noise);
    let dt = 0.01;
    let state = RefLooseState {
        q_es: [1.0, 0.0, 0.0, 0.0],
        pos_e: [WGS84_A, 0.0, 0.0],
        vel_e: [5.0, -1.0, 0.2],
        b_w: [0.0, 0.0, 0.0],
        b_f: [0.0, 0.0, 0.0],
        s_w: [1.0, 1.0, 1.0],
        s_f: [1.0, 1.0, 1.0],
    };
    let omega_1 = [0.0, 0.15, -0.05];
    let omega_2 = [0.05, 0.25, 0.10];
    let accel_1 = [9.4, 0.2, 0.1];
    let accel_2 = [10.2, -0.1, -0.2];
    let expected = ref_two_sample_heun_step(state, omega_1, omega_2, accel_1, accel_2, dt);
    loose.init_from_reference_ecef_state(
        state.q_es,
        [state.pos_e[0] as f64, state.pos_e[1] as f64, state.pos_e[2] as f64],
        state.vel_e,
        state.b_w,
        state.b_f,
        state.s_w,
        state.s_f,
        [1.0, 0.0, 0.0, 0.0],
        None,
    );
    loose.predict(CLooseImuDelta {
        dax_1: omega_1[0] * dt,
        day_1: omega_1[1] * dt,
        daz_1: omega_1[2] * dt,
        dvx_1: accel_1[0] * dt,
        dvy_1: accel_1[1] * dt,
        dvz_1: accel_1[2] * dt,
        dax_2: omega_2[0] * dt,
        day_2: omega_2[1] * dt,
        daz_2: omega_2[2] * dt,
        dvx_2: accel_2[0] * dt,
        dvy_2: accel_2[1] * dt,
        dvz_2: accel_2[2] * dt,
        dt,
    });

    let x = loose.nominal();
    assert_close(x.q0, expected.q_es[0], 1.0e-5, "varying q0");
    assert_close(x.q1, expected.q_es[1], 1.0e-5, "varying q1");
    assert_close(x.q2, expected.q_es[2], 1.0e-5, "varying q2");
    assert_close(x.q3, expected.q_es[3], 1.0e-5, "varying q3");
    assert_close(x.pn, expected.pos_e[0], 1.0e-3, "varying pn");
    assert_close(x.pe, expected.pos_e[1], 1.0e-4, "varying pe");
    assert_close(x.pd, expected.pos_e[2], 1.0e-4, "varying pd");
    assert_close(x.vn, expected.vel_e[0], 1.0e-4, "varying vn");
    assert_close(x.ve, expected.vel_e[1], 1.0e-4, "varying ve");
    assert_close(x.vd, expected.vel_e[2], 1.0e-4, "varying vd");
}

#[test]
fn loose_predict_includes_vertical_specific_force_in_velocity_update() {
    let noise = LoosePredictNoise::reference_nsr_demo();
    let mut loose = CLooseWrapper::new(noise);
    let dt = 0.02;
    let state = RefLooseState {
        q_es: [1.0, 0.0, 0.0, 0.0],
        pos_e: [WGS84_A, 0.0, 0.0],
        vel_e: [0.0, 0.0, 0.0],
        b_w: [0.0, 0.0, 0.0],
        b_f: [0.0, 0.0, 0.0],
        s_w: [1.0, 1.0, 1.0],
        s_f: [1.0, 1.0, 1.0],
    };
    let accel_1 = [0.0, 0.0, 5.0];
    let accel_2 = [0.0, 0.0, 7.0];
    let expected = ref_two_sample_heun_step(state, [0.0; 3], [0.0; 3], accel_1, accel_2, dt);
    loose.init_from_reference_ecef_state(
        state.q_es,
        [state.pos_e[0] as f64, state.pos_e[1] as f64, state.pos_e[2] as f64],
        state.vel_e,
        state.b_w,
        state.b_f,
        state.s_w,
        state.s_f,
        [1.0, 0.0, 0.0, 0.0],
        None,
    );
    loose.predict(CLooseImuDelta {
        dax_1: 0.0,
        day_1: 0.0,
        daz_1: 0.0,
        dvx_1: accel_1[0] * dt,
        dvy_1: accel_1[1] * dt,
        dvz_1: accel_1[2] * dt,
        dax_2: 0.0,
        day_2: 0.0,
        daz_2: 0.0,
        dvx_2: accel_2[0] * dt,
        dvy_2: accel_2[1] * dt,
        dvz_2: accel_2[2] * dt,
        dt,
    });
    assert_close(loose.nominal().vd, expected.vel_e[2], 1.0e-5, "vertical specific force");
}

#[test]
fn loose_reference_batch_matches_gps_only_reference_update() {
    let noise = LoosePredictNoise::default();
    let p_diag = core::array::from_fn(|i| 0.02 * (i as f32 + 1.0));
    let q_es = quat_normalize([0.93, 0.07, -0.12, 0.33]);
    let q_cs = quat_normalize([0.9997, 0.015, -0.01, 0.018]);
    let pos_ecef = [WGS84_A - 12.0, 34.0, 56.0];
    let vel_ecef = [4.0, -1.5, 0.25];
    let gyro_bias = [0.01, -0.02, 0.03];
    let accel_bias = [0.15, -0.12, 0.08];
    let gyro_scale = [1.001, 0.999, 1.0005];
    let accel_scale = [0.998, 1.002, 1.001];
    let pos_meas = [
        (pos_ecef[0] + 2.5) as f64,
        (pos_ecef[1] - 1.25) as f64,
        (pos_ecef[2] + 0.8) as f64,
    ];
    let gyro_radps = [0.2, -0.1, 0.05];
    let accel_mps2 = [2.5, 0.0, 9.81];

    let mut gps_only = CLooseWrapper::new(noise);
    gps_only.init_from_reference_ecef_state(
        q_es,
        [pos_ecef[0] as f64, pos_ecef[1] as f64, pos_ecef[2] as f64],
        vel_ecef,
        gyro_bias,
        accel_bias,
        gyro_scale,
        accel_scale,
        q_cs,
        Some(p_diag),
    );
    let mut batch = CLooseWrapper::new(noise);
    batch.init_from_reference_ecef_state(
        q_es,
        [pos_ecef[0] as f64, pos_ecef[1] as f64, pos_ecef[2] as f64],
        vel_ecef,
        gyro_bias,
        accel_bias,
        gyro_scale,
        accel_scale,
        q_cs,
        Some(p_diag),
    );

    gps_only.fuse_gps_reference(pos_meas, None, 1.8, 0.0, 0.5);
    batch.fuse_reference_batch(Some(pos_meas), None, 1.8, 0.0, 0.5, gyro_radps, accel_mps2, 0.01);

    assert_nominal_close(batch.nominal(), gps_only.nominal(), 1.0e-6, "gps_only_batch");
    let batch_p = batch.covariance();
    let gps_p = gps_only.covariance();
    assert_covariance_close(&batch_p, &gps_p, 1.0e-6, "gps_only_batch");
}

#[test]
fn loose_reference_batch_matches_nhc_only_reference_update() {
    let noise = LoosePredictNoise::default();
    let p_diag = core::array::from_fn(|i| 0.03 * (i as f32 + 1.0));
    let q_es = quat_normalize([0.96, 0.08, 0.02, -0.25]);
    let q_cs = quat_normalize([0.9998, -0.01, 0.015, -0.005]);
    let pos_ecef = [WGS84_A - 30.0, 120.0, -85.0];
    let vel_ecef = [6.0, -2.0, 0.4];
    let gyro_bias = [0.0, 0.0, 0.0];
    let accel_bias = [0.0, 0.0, 0.0];
    let gyro_scale = [1.0, 1.0, 1.0];
    let accel_scale = [1.0, 1.0, 1.0];
    let gyro_radps = [0.002, -0.003, 0.001];
    let accel_mps2 = [0.05, -0.02, 9.809];

    let mut nhc_only = CLooseWrapper::new(noise);
    nhc_only.init_from_reference_ecef_state(
        q_es,
        [pos_ecef[0] as f64, pos_ecef[1] as f64, pos_ecef[2] as f64],
        vel_ecef,
        gyro_bias,
        accel_bias,
        gyro_scale,
        accel_scale,
        q_cs,
        Some(p_diag),
    );
    let mut batch = CLooseWrapper::new(noise);
    batch.init_from_reference_ecef_state(
        q_es,
        [pos_ecef[0] as f64, pos_ecef[1] as f64, pos_ecef[2] as f64],
        vel_ecef,
        gyro_bias,
        accel_bias,
        gyro_scale,
        accel_scale,
        q_cs,
        Some(p_diag),
    );

    nhc_only.fuse_nhc_reference(gyro_radps, accel_mps2, 0.01);
    batch.fuse_reference_batch(None, None, 0.0, 0.0, 1.0, gyro_radps, accel_mps2, 0.01);

    assert_nominal_close(batch.nominal(), nhc_only.nominal(), 1.0e-6, "nhc_only_batch");
    let batch_p = batch.covariance();
    let nhc_p = nhc_only.covariance();
    assert_covariance_close(&batch_p, &nhc_p, 1.0e-6, "nhc_only_batch");
}

#[test]
#[ignore = "legacy f32 helper is looser than the exact MATLAB-checked reference path"]
fn loose_reference_batch_matches_reference_combined_gps_nhc_update() {
    let noise = LoosePredictNoise::default();
    let p_diag = core::array::from_fn(|i| 0.015 * (i as f32 + 1.0));
    let q_es = quat_normalize([0.97, 0.06, -0.08, 0.21]);
    let q_cs = quat_normalize([0.9997, 0.012, -0.018, 0.01]);
    let pos_ecef = [WGS84_A - 80.0, 150.0, 42.0];
    let vel_ecef = [8.5, -2.2, 0.3];
    let gyro_bias = [0.004, -0.003, 0.002];
    let accel_bias = [0.08, -0.05, 0.03];
    let gyro_scale = [1.001, 0.999, 1.0002];
    let accel_scale = [0.999, 1.0015, 1.0005];
    let pos_meas = [
        (pos_ecef[0] + 1.2) as f64,
        (pos_ecef[1] - 0.7) as f64,
        (pos_ecef[2] + 1.8) as f64,
    ];
    let gyro_radps = [0.002, -0.001, 0.0015];
    let accel_mps2 = [0.03, -0.01, 9.809];

    let mut batch = CLooseWrapper::new(noise);
    batch.init_from_reference_ecef_state(
        q_es,
        [pos_ecef[0] as f64, pos_ecef[1] as f64, pos_ecef[2] as f64],
        vel_ecef,
        gyro_bias,
        accel_bias,
        gyro_scale,
        accel_scale,
        q_cs,
        Some(p_diag),
    );

    let x_ref = RefLooseFullState {
        q_es,
        pos_e: pos_ecef,
        vel_e: vel_ecef,
        b_f: accel_bias,
        b_w: gyro_bias,
        s_f: accel_scale,
        s_w: gyro_scale,
        q_cs,
    };
    let mut p_ref = [[0.0_f32; 24]; 24];
    for i in 0..24 {
        p_ref[i][i] = p_diag[i];
    }
    let (x_expected, p_expected) = reference_batch_update(
        x_ref,
        p_ref,
        Some([pos_meas[0] as f32, pos_meas[1] as f32, pos_meas[2] as f32]),
        1.4,
        0.4,
        gyro_radps,
        accel_mps2,
        0.01,
    );

    batch.fuse_reference_batch(Some(pos_meas), None, 1.4, 0.0, 0.4, gyro_radps, accel_mps2, 0.01);

    let x_actual = batch.nominal();
    assert_close(x_actual.q0, x_expected.q_es[0], 2.0e-4, "combined.q0");
    assert_close(x_actual.q1, x_expected.q_es[1], 2.0e-4, "combined.q1");
    assert_close(x_actual.q2, x_expected.q_es[2], 2.0e-4, "combined.q2");
    assert_close(x_actual.q3, x_expected.q_es[3], 2.0e-4, "combined.q3");
    assert_close(x_actual.pn, x_expected.pos_e[0], 2.0e-4, "combined.pn");
    assert_close(x_actual.pe, x_expected.pos_e[1], 2.0e-4, "combined.pe");
    assert_close(x_actual.pd, x_expected.pos_e[2], 2.0e-4, "combined.pd");
    assert_close(x_actual.vn, x_expected.vel_e[0], 2.0e-4, "combined.vn");
    assert_close(x_actual.ve, x_expected.vel_e[1], 2.0e-4, "combined.ve");
    assert_close(x_actual.vd, x_expected.vel_e[2], 2.0e-4, "combined.vd");
    assert_close(x_actual.bax, x_expected.b_f[0], 2.0e-4, "combined.bax");
    assert_close(x_actual.bay, x_expected.b_f[1], 2.0e-4, "combined.bay");
    assert_close(x_actual.baz, x_expected.b_f[2], 2.0e-4, "combined.baz");
    assert_close(x_actual.bgx, x_expected.b_w[0], 2.0e-4, "combined.bgx");
    assert_close(x_actual.bgy, x_expected.b_w[1], 2.0e-4, "combined.bgy");
    assert_close(x_actual.bgz, x_expected.b_w[2], 2.0e-4, "combined.bgz");
    assert_close(x_actual.sax, x_expected.s_f[0], 2.0e-4, "combined.sax");
    assert_close(x_actual.say, x_expected.s_f[1], 2.0e-4, "combined.say");
    assert_close(x_actual.saz, x_expected.s_f[2], 2.0e-4, "combined.saz");
    assert_close(x_actual.sgx, x_expected.s_w[0], 2.0e-4, "combined.sgx");
    assert_close(x_actual.sgy, x_expected.s_w[1], 2.0e-4, "combined.sgy");
    assert_close(x_actual.sgz, x_expected.s_w[2], 2.0e-4, "combined.sgz");
    assert_close(x_actual.qcs0, x_expected.q_cs[0], 2.0e-4, "combined.qcs0");
    assert_close(x_actual.qcs1, x_expected.q_cs[1], 2.0e-4, "combined.qcs1");
    assert_close(x_actual.qcs2, x_expected.q_cs[2], 2.0e-4, "combined.qcs2");
    assert_close(x_actual.qcs3, x_expected.q_cs[3], 2.0e-4, "combined.qcs3");
    let p_actual = batch.covariance();
    assert_covariance_close(&p_actual, &p_expected, 2.0e-4, "combined_batch");
}

#[test]
fn loose_error_transition_matches_reference_formula() {
    let noise = LoosePredictNoise::reference_nsr_demo();
    let q_es = quat_normalize([0.95, 0.1, -0.15, 0.22]);
    let q_cs = quat_normalize([0.999, 0.01, 0.02, -0.03]);
    let pos_ecef = [WGS84_A - 100.0, 50.0, -20.0];
    let vel_ecef = [7.0, -1.0, 0.5];
    let gyro_bias = [0.01, -0.02, 0.03];
    let accel_bias = [0.1, -0.2, 0.3];
    let gyro_scale = [1.001, 0.999, 1.0005];
    let accel_scale = [0.998, 1.002, 1.001];
    let dt = 0.02;
    let imu = CLooseImuDelta {
        dax_1: 0.001,
        day_1: -0.002,
        daz_1: 0.0005,
        dvx_1: 0.15,
        dvy_1: -0.03,
        dvz_1: 0.19,
        dax_2: 0.003,
        day_2: -0.001,
        daz_2: 0.0008,
        dvx_2: 0.18,
        dvy_2: -0.01,
        dvz_2: 0.21,
        dt,
    };
    let mut loose = CLooseWrapper::new(noise);
    loose.init_from_reference_ecef_state(
        q_es,
        [pos_ecef[0] as f64, pos_ecef[1] as f64, pos_ecef[2] as f64],
        vel_ecef,
        gyro_bias,
        accel_bias,
        gyro_scale,
        accel_scale,
        q_cs,
        None,
    );
    let (f_actual, g_actual) = loose.compute_error_transition(imu);
    let accel_raw = [imu.dvx_2 / dt, imu.dvy_2 / dt, imu.dvz_2 / dt];
    let gyro_raw = [imu.dax_2 / dt, imu.day_2 / dt, imu.daz_2 / dt];
    let (f_expected, g_expected) = reference_transition(
        q_es,
        accel_bias,
        gyro_bias,
        accel_scale,
        gyro_scale,
        accel_raw,
        gyro_raw,
        dt,
    );
    assert_covariance_close(&f_actual, &f_expected, 1.0e-5, "transition_F");
    for i in 0..24 {
        for j in 0..21 {
            assert_close(g_actual[i][j], g_expected[i][j], 1.0e-5, &format!("transition_G[{i}][{j}]"));
        }
    }
}

#[test]
fn loose_predict_covariance_matches_reference_phi_p_phi_t_plus_q() {
    let noise = LoosePredictNoise::reference_nsr_demo();
    let q_es = quat_normalize([0.95, 0.1, -0.15, 0.22]);
    let q_cs = quat_normalize([0.999, 0.01, 0.02, -0.03]);
    let pos_ecef = [WGS84_A - 100.0, 50.0, -20.0];
    let vel_ecef = [7.0, -1.0, 0.5];
    let gyro_bias = [0.01, -0.02, 0.03];
    let accel_bias = [0.1, -0.2, 0.3];
    let gyro_scale = [1.001, 0.999, 1.0005];
    let accel_scale = [0.998, 1.002, 1.001];
    let dt = 0.02;
    let imu = CLooseImuDelta {
        dax_1: 0.001,
        day_1: -0.002,
        daz_1: 0.0005,
        dvx_1: 0.15,
        dvy_1: -0.03,
        dvz_1: 0.19,
        dax_2: 0.003,
        day_2: -0.001,
        daz_2: 0.0008,
        dvx_2: 0.18,
        dvy_2: -0.01,
        dvz_2: 0.21,
        dt,
    };
    let mut loose = CLooseWrapper::new(noise);
    loose.init_from_reference_ecef_state(
        q_es,
        [pos_ecef[0] as f64, pos_ecef[1] as f64, pos_ecef[2] as f64],
        vel_ecef,
        gyro_bias,
        accel_bias,
        gyro_scale,
        accel_scale,
        q_cs,
        None,
    );
    let mut p0 = [[0.0_f32; 24]; 24];
    for i in 0..24 {
        for j in 0..24 {
            p0[i][j] = if i == j {
                0.01 * (i as f32 + 1.0)
            } else if (i as isize - j as isize).abs() == 1 {
                1.0e-3 * (i.min(j) as f32 + 1.0)
            } else {
                0.0
            };
        }
    }
    loose.set_covariance(p0);

    let accel_raw = [imu.dvx_2 / dt, imu.dvy_2 / dt, imu.dvz_2 / dt];
    let gyro_raw = [imu.dax_2 / dt, imu.day_2 / dt, imu.daz_2 / dt];
    let (phi, g) = reference_transition(
        q_es,
        accel_bias,
        gyro_bias,
        accel_scale,
        gyro_scale,
        accel_raw,
        gyro_raw,
        dt,
    );
    let q_diag = [
        noise.gyro_var * dt,
        noise.gyro_var * dt,
        noise.gyro_var * dt,
        noise.accel_var * dt,
        noise.accel_var * dt,
        noise.accel_var * dt,
        noise.accel_bias_rw_var * dt,
        noise.accel_bias_rw_var * dt,
        noise.accel_bias_rw_var * dt,
        noise.gyro_bias_rw_var * dt,
        noise.gyro_bias_rw_var * dt,
        noise.gyro_bias_rw_var * dt,
        noise.accel_scale_rw_var * dt,
        noise.accel_scale_rw_var * dt,
        noise.accel_scale_rw_var * dt,
        noise.gyro_scale_rw_var * dt,
        noise.gyro_scale_rw_var * dt,
        noise.gyro_scale_rw_var * dt,
        noise.mount_align_rw_var * dt,
        noise.mount_align_rw_var * dt,
        noise.mount_align_rw_var * dt,
    ];
    let phi_p = mat24_mul(&phi, &p0);
    let expected = mat24_mul(&phi_p, &mat24_t(&phi));
    let q_term = mat24x21_diag_mat21x24(&g, q_diag);
    let mut expected_p = [[0.0_f32; 24]; 24];
    for i in 0..24 {
        for j in 0..24 {
            expected_p[i][j] = expected[i][j] + q_term[i][j];
        }
    }

    loose.predict(imu);
    let actual_p = loose.covariance();
    assert_covariance_close(&actual_p, &expected_p, 2.0e-4, "predict_covariance");
}
