# Frames And Quaternions

The project uses active rotations and scalar-first quaternions:

$$
q_{ab} =
\begin{bmatrix}
q_w & q_x & q_y & q_z
\end{bmatrix}^{T}.
$$

\(C_{ab}\) maps coordinates from frame \(b\) to frame \(a\):

$$
\begin{aligned}
x_a &= C_{ab} x_b,\\
R(q_{ab}) &= C_{ab},\\
R(q_1 q_2) &= R(q_1)R(q_2).
\end{aligned}
$$

For a normalized scalar-first quaternion, the rotation matrix used by the Rust
runtime is:

$$
C_{ab} = R(q_{ab}) =
\begin{bmatrix}
1 - 2(q_y^2 + q_z^2) &
2(q_x q_y - q_w q_z) &
2(q_x q_z + q_w q_y) \\
2(q_x q_y + q_w q_z) &
1 - 2(q_x^2 + q_z^2) &
2(q_y q_z - q_w q_x) \\
2(q_x q_z - q_w q_y) &
2(q_y q_z + q_w q_x) &
1 - 2(q_x^2 + q_y^2)
\end{bmatrix}.
$$

Frame names:

| Symbol | Meaning |
| --- | --- |
| `b` | raw IMU body/sensor frame |
| `v` | vehicle frame, forward-right-down |
| `n` | local NED navigation frame |
| `e` | ECEF frame |

The public mount quaternion is the physical vehicle-to-body mount:

$$
\begin{aligned}
q_{bv},\qquad
x_b &= C_{bv} x_v,\\
C_{vb} &= C_{bv}^{\top},\\
x_v &= C_{vb} x_b .
\end{aligned}
$$

The EKF attitude is:

$$
q_{nv},\qquad x_n = C_{nv} x_v.
$$

## Error Injection

The EKF uses a nominal state plus a local error state. A small-angle vector
`delta theta` is converted to:

$$
\delta q(\delta\theta) \approx
\begin{bmatrix}
1 & \frac{1}{2}\delta\theta_x & \frac{1}{2}\delta\theta_y & \frac{1}{2}\delta\theta_z
\end{bmatrix}^T .
$$

Runtime attitude and residual mount errors are injected on different sides:

$$
\begin{aligned}
q_{nv}^+ &= q_{nv}\,\delta q(\delta\theta_v),\\
q_{bv}^+ &= \delta q(\delta\psi_{bv})\,q_{bv}.
\end{aligned}
$$

The standalone align filter also left-multiplies generic small-angle mount
updates into \(q_{bv}\). Its horizontal-acceleration yaw update is the exception:
it right-multiplies a vehicle-yaw correction, because that update is formulated
as an angle between horizontal vehicle-frame acceleration vectors.

This convention is enforced by coordinate-convention tests in `sensor_fusion`.
