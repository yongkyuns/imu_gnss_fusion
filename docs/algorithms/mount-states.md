# Mount States And Navigation Attitude

This page describes the runtime mount state, the separate align stage used in
automatic mount mode, and the difference between attitude-only filters and the
implemented GNSS-aided navigation filter.

For the exact frame convention, see [](frames.md). For the implemented EKF
equations, see [](runtime-ekf.md).

## What The Mount State Represents

The IMU reports angular rate and specific force in the raw body frame $b$. The
vehicle motion model is written in the vehicle frame $v$, where $x$ points
forward, $y$ points right, and $z$ points down. The physical mount is therefore
the rotation between those frames:

$$
x_b = C_{bv}x_v,\qquad x_v = C_{vb}x_b,\qquad C_{vb}=C_{bv}^T .
$$

The EKF stores this mount as $q_{bv}$. It also stores the vehicle attitude
$q_{nv}$, which maps vehicle-frame vectors into the local navigation frame:

$$
x_n = C_{nv}x_v .
$$

Together, these two rotations define the sensor-body attitude in navigation
coordinates:

$$
x_n = C_{nv}C_{vb}x_b .
$$

The factorization separates vehicle attitude from sensor mounting. NHC, vehicle
speed, road-event features, and vehicle acceleration are expressed in the
vehicle frame, not in the sensor body frame.

## Why Mount Is Part Of The EKF

Mount error changes the propagation model itself. For each IMU interval, the
runtime forms body-frame increments and rotates them through the current mount:

$$
\Delta\alpha_v = C_{vb}(\Delta\alpha_b-b_g\Delta t),
\qquad
\Delta v_v = C_{vb}(\Delta v_b-b_a\Delta t).
$$

Those vehicle-frame increments then propagate attitude, velocity, and position:

$$
q_{nv}^+ = q_{nv}\delta q(\Delta\alpha_v),
\qquad
v_n^+ = v_n + C_{nv}\Delta v_v + g_n\Delta t .
$$

Mount error changes the mechanization. A forward acceleration can project into
lateral or vertical axes, and a yaw-rate increment can project into roll or
pitch axes. If mount were treated only as a preprocessing constant, later
residuals would be absorbed by attitude, velocity, position, or bias states even
when the underlying error is sensor-to-vehicle rotation.

The implemented nominal EKF state is:

$$
x =
\begin{bmatrix}
q_{nv} & v_n^T & p_n^T & b_g^T & b_a^T & q_{bv}
\end{bmatrix}^T .
$$

The 18-dimensional error state is:

$$
\delta x =
\begin{bmatrix}
\delta\theta_v &
\delta v_n &
\delta p_n &
\delta b_g &
\delta b_a &
\delta\psi_{bv}
\end{bmatrix}^T .
$$

Biases are additive corrections in the raw body frame. Mount errors are local
small-angle residuals on the physical mount quaternion. In automatic mount mode,
align seeds $q_{bv}$ and the EKF continues estimating residual mount states. In
manual mount mode, the caller supplies the seed and the EKF starts with a tight
mount prior rather than running align.

## Why Align Is Separate

The EKF is a local error-state filter. Its propagation and measurement
Jacobians describe small perturbations around the current nominal state. The
linearized model assumes the nominal attitude and mount are close enough that
small-angle corrections are valid.

Automatic mount mode therefore uses `align` as a reduced estimator before EKF
initialization. Align estimates only $q_{bv}$ and a 3 by 3 local mount
covariance. It uses broader-capture cues:

- static accelerometer tilt samples to seed and refine roll/pitch;
- GNSS-derived horizontal acceleration to constrain yaw;
- planar turn-gyro consistency to refine roll/pitch during turns.

Align does not run NHC and does not estimate vehicle position, velocity,
navigation attitude, or IMU biases. It provides an initial physical mount seed
for the local EKF. The split is particularly relevant for yaw, because gravity
does not observe yaw.

The handoff covariance is the covariance of the reduced align model. It is not
an independent validation that mount, vehicle attitude, road bank, and IMU bias
have been separated.

## How Mount And Attitude Ambiguity Arises

Because the EKF estimates both $q_{nv}$ and $q_{bv}$, some physical effects can
be explained by more than one state split. The total sensor-body attitude is
approximately the product of vehicle attitude and inverse mount:

$$
C_{nb}=C_{nv}C_{vb}.
$$

For roll about the vehicle forward axis, a change in vehicle roll can often be
offset by a compensating change in mount roll while preserving similar
body-frame signals. GNSS and NHC constrain vehicle motion, but they do not by
themselves identify how a forward-axis rotation should be split between
road/vehicle roll and physical mount roll.

The implemented instantaneous NHC rows observe vehicle-frame lateral and
vertical velocity:

$$
e_y^T C_{nv}^T v_n \approx 0,\qquad
e_z^T C_{nv}^T v_n \approx 0 .
$$

Their direct measurement Jacobian has attitude and velocity columns, not direct
residual-mount columns. Mount can still move through Kalman gain when
propagation has created cross-covariance between mount and observed states, but
that is covariance-mediated correction. It is not the same as NHC directly
measuring mount roll.

This distinction matters on banked roads. A sustained bank can be inconsistent
with a flat-road vehicle-roll prior. The prior then pushes vehicle roll toward
zero, and the remaining rotation can be absorbed into mount. The prior is an
assumption on the vehicle/road roll channel, not an independent observation of
mount roll.

## AHRS Filters Versus Full Navigation

Mahony- and Madgwick-style filters are attitude filters. In 6DOF form they
combine gyro integration with accelerometer leveling. In 9DOF form they also
use a magnetometer or another heading cue. They are appropriate when the target
state is attitude and the accelerometer is a sufficiently reliable gravity
reference.

That assumption is fragile in a vehicle. An accelerometer measures specific
force, not gravity alone. During acceleration, braking, cornering, grade
changes, bumps, and suspension motion, the measured vector is:

$$
f_b \approx C_{bn}(a_n-g_n) + b_a + n_a .
$$

An attitude-only filter that treats $f_b$ as a gravity direction has no velocity
or position state to represent translational acceleration. Sustained
non-gravitational acceleration can therefore produce attitude error:

| Vehicle motion | Common attitude-only failure mode |
| --- | --- |
| Acceleration | forward specific force can bias pitch |
| Braking | opposite pitch bias |
| Cornering | lateral specific force can bias roll |
| Uphill/downhill grade | gravity projection and acceleration are hard to separate |
| Bumps and suspension motion | transient vertical force perturbs tilt estimates |

A magnetometer can help yaw in a clean magnetic environment, but it does not
solve acceleration-as-gravity contamination. In passenger vehicles, phone and
board magnetometers are also often disturbed by the cabin, electronics, mounts,
and nearby ferromagnetic material.

The navigation EKF models the same IMU samples differently. The accelerometer is
used as a propagation input for velocity and position, not primarily as a direct
attitude measurement during dynamic motion. GNSS position and velocity then
constrain the propagated navigation state:

$$
r_p = p_\mathrm{gnss}-p_n,\qquad
r_v = v_\mathrm{gnss}-v_n .
$$

This allows acceleration and braking to be represented as changes in velocity
and position, and cornering to be represented as lateral acceleration and yaw
rate. The result remains limited by GNSS quality, observability, bias modeling,
and vehicle assumptions.

## Biases Need Navigation Context

Gyro bias causes attitude drift. Accelerometer bias causes velocity and position
drift, and can mimic small tilt or mount errors. A pure attitude filter can
estimate gyro bias through attitude feedback, but it generally has no external
velocity/position residual that can identify accelerometer bias as an inertial
navigation error.

This EKF estimates additive raw-body gyro and accelerometer biases:

$$
\omega_b^\mathrm{corr}=\omega_b-b_g,\qquad
f_b^\mathrm{corr}=f_b-b_a .
$$

GNSS velocity/position, NHC, stationary periods, and motion excitation allocate
residuals among attitude, velocity, bias, and mount. Bias and mount can still
trade in weak motion regimes. Without bias states, persistent sensor offsets
must be absorbed by attitude, mount, or velocity errors.

## Why Scale Factors Are Omitted

Some INS formulations estimate accelerometer and gyro scale factors. The
current filter omits scale states and estimates only additive biases. The
assumption is:

- consumer IMU scale-factor errors are typically smaller than bias, mount,
  vibration, timestamp, and GNSS-quality effects over short vehicle trips;
- adding scale states increases state dimension and weakens observability unless
  the dataset contains rich multi-axis excitation;
- scale-factor errors over typical phone/embedded replay intervals are below
  the current modeling error floor.

Scale factors are not assumed to be physically zero. They are omitted from this
state because the current measurement set is intended to identify navigation
state, additive body-frame biases, and mount.

## Why Lever Arm Is Omitted

The runtime fuses GNSS position and velocity directly against the navigation
state:

$$
p_\mathrm{gnss}\approx p_n,\qquad v_\mathrm{gnss}\approx v_n .
$$

There is no explicit GNSS-antenna-to-IMU lever-arm state or correction. The
assumption is that the GNSS solution and IMU are colocated at the accuracy level
targeted by the current phone/mobile and compact embedded use cases. A remote
antenna, high-grade GNSS, or large rotational rates require explicit lever-arm
modeling.

## Practical Implications

- AHRS-style filters estimate attitude; they do not estimate the vehicle
  navigation state or sensor-to-vehicle mount.
- The navigation EKF estimates attitude, velocity, position, additive IMU
  biases, and mount in one state.
- Align is an initialization estimator for automatic mount mode.
- EKF mount covariance is model covariance, not evidence that road bank, vehicle
  roll, and mount roll have been uniquely separated.
- The flat-road vehicle-roll prior is an environmental assumption.

## References

- [Mahony et al., Nonlinear Complementary Filters on the Special Orthogonal Group](https://ieeexplore.ieee.org/document/4608934)
- [Madgwick, An efficient orientation filter for inertial and inertial/magnetic sensor arrays](https://x-io.co.uk/downloads/madgwick_internal_report.pdf)
- [OpenIMU AHRS/INS documentation](https://openimu.readthedocs.io/en/latest/)
- [](../reference/prior-work.md)
