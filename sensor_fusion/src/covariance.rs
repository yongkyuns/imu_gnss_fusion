//! Shared covariance matrix helpers for generated EKF models.

#[derive(Clone, Copy, Debug)]
pub(crate) struct SparseCovariancePolicy {
    pub skip_zero_f_i: bool,
    pub skip_zero_f_j: bool,
    pub skip_zero_g_i: bool,
    pub skip_zero_g_j: bool,
    pub skip_zero_q: bool,
}

impl SparseCovariancePolicy {
    pub(crate) const EKF: Self = Self {
        skip_zero_f_i: true,
        skip_zero_f_j: true,
        skip_zero_g_i: true,
        skip_zero_g_j: true,
        skip_zero_q: false,
    };
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(crate) fn predict_sparse<const N: usize, const M: usize, const FW: usize, const GW: usize>(
    f: &[[f32; N]; N],
    g: &[[f32; M]; N],
    p: &[[f32; N]; N],
    q: &[f32; M],
    f_row_counts: &[usize; N],
    f_row_cols: &[[usize; FW]; N],
    g_row_counts: &[usize; N],
    g_row_cols: &[[usize; GW]; N],
    policy: SparseCovariancePolicy,
) -> [[f32; N]; N] {
    let mut next = [[0.0; N]; N];
    for i in 0..N {
        for j in i..N {
            let mut accum = 0.0;
            for ia in 0..f_row_counts[i] {
                let a = f_row_cols[i][ia];
                let fi = f[i][a];
                if policy.skip_zero_f_i && fi == 0.0 {
                    continue;
                }
                for jb in 0..f_row_counts[j] {
                    let b = f_row_cols[j][jb];
                    let fj = f[j][b];
                    if !policy.skip_zero_f_j || fj != 0.0 {
                        accum += fi * p[a][b] * fj;
                    }
                }
            }
            for ia in 0..g_row_counts[i] {
                let a = g_row_cols[i][ia];
                let gi = g[i][a];
                if policy.skip_zero_g_i && gi == 0.0 {
                    continue;
                }
                if policy.skip_zero_q && q[a] == 0.0 {
                    continue;
                }
                for jb in 0..g_row_counts[j] {
                    let b = g_row_cols[j][jb];
                    if a == b {
                        let gj = g[j][b];
                        if !policy.skip_zero_g_j || gj != 0.0 {
                            accum += gi * q[a] * gj;
                        }
                    }
                }
            }
            next[i][j] = accum;
            next[j][i] = accum;
        }
    }
    next
}

pub(crate) fn symmetrize<const N: usize>(p: &mut [[f32; N]; N]) {
    for i in 0..N {
        for j in (i + 1)..N {
            let sym = 0.5 * (p[i][j] + p[j][i]);
            p[i][j] = sym;
            p[j][i] = sym;
        }
    }
}
