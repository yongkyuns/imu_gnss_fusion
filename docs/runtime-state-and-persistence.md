# Runtime State And Persistence

`SensorFusion` has two different kinds of persistence. Keeping them separate avoids most confusion:

- **In-memory EKF continuity** means the same `SensorFusion` object keeps its current navigation state, covariance, mount estimate, IMU biases, diagnostics, and timing state across a normal stream pause or MCU sleep.
- **Saved priors** are calibration values the caller may store outside the library and provide to a later fresh context. The library reports when those priors are worth saving, but it does not write persistent storage itself.

Today the iOS app persists mount memory when fusion is stable. It does not serialize a complete EKF snapshot. Embedded firmware can keep the full context in retained memory across sleep; if the process or MCU loses that memory, start a fresh context and seed it with any saved priors that were marked stable.

```{figure} _static/diagrams/ekf-persistence-lifecycle-elk.svg
:alt: EKF runtime state and persistence lifecycle through initialization, running, pauses, sleep classes, and GNSS reseed.
:class: framed

Normal stream pauses preserve the `SensorFusion` context. Sleep gaps are classified by the next IMU timestamp; source switches, replay changes, and true fresh starts create a new context.
```

## Public State

Every input update returns one public lifecycle state:

| State | Meaning | Navigation usable? |
| --- | --- | --- |
| `NotReady` | no usable filter state exists yet | no |
| `Initializing` | inputs are arriving, but mount or EKF initialization is still in progress | no |
| `Running` | EKF navigation is available, but not stable enough for saving priors | yes |
| `Stable` | navigation is available and invariant states are stable enough to persist | yes |
| `Degraded` | navigation exists, but diagnostics report unhealthy input/state conditions | usually yes unless `navigation_usable` is false |
| `DegradedDeadReckoning` | navigation is continuing after a medium sleep gap without new GNSS yet | yes |
| `AwaitingGnssReseed` | mount and bias priors are retained, but navigation must be reseeded from GNSS | no |

Use `Update.state` and `Update.navigation_usable` after each sample. `SensorFusion::health()` returns the same state plus convergence metrics and reason bits. The boolean flags are intentionally derived from the single state:

- `navigation_usable`: public pose/velocity output can be consumed.
- `navigation_started`: navigation became usable on this update.
- `stable`: priors are mature enough to save.
- `degraded`: the current state or inputs are unhealthy or post-gap.

Do not separately infer health from whether an EKF object exists internally. During reseed, the runtime may retain calibration and covariance while public navigation is intentionally unavailable.

## What Callers Should Do

For normal live operation:

- Keep the same `SensorFusion` object across stop/start streaming, app backgrounding, and short MCU sleep when the device is physically stationary.
- Continue feeding timestamped IMU and GNSS samples. The library classifies short, medium, and long sleep gaps from sample timestamps.
- Consume pose, velocity, and vehicle-frame outputs only when `navigation_usable` is true.
- Save external priors only when `health.stable` is true.

Start a fresh context when the data source changes or the physical situation invalidates continuity:

- replaying a different dataset;
- switching from replay to live capture;
- changing the phone/device mount;
- losing retained EKF memory;
- allowing the device or vehicle to move while samples are stopped.

The sleep model assumes the device is stationary during the missing sample interval. If that assumption is false, preserving the context can make the old position, velocity, and yaw overconfident. Reset and reseed from GNSS instead.

## Stable Prior Criteria

`Stable` is intentionally stricter than `Running`. Running means the navigation output can be used now; stable means the slow states are mature enough to become saved priors for a future fresh context.

The diagnostic module evaluates stability from accumulated post-initialization motion plus a recent tail window of mount and bias estimates:

| Criterion | Current gate | Reason |
| --- | --- | --- |
| post-initialization time | `>= 180 s` | avoid saving the early alignment transient |
| driven distance | `>= 750 m` | require enough vehicle excitation to separate mount, attitude, and bias effects |
| recent tail duration | `>= 90 s` | evaluate invariants over a fresh window, not only whole-trip averages |
| recent tail samples | `>= 30` | avoid declaring stability from sparse updates |
| mount tail drift | `<= 0.50 deg` | reject slowly moving mount estimates |
| mount tail standard deviation | `<= 0.35 deg` | reject noisy mount estimates |
| gyro-bias tail drift | `<= 0.00035 rad/s` | reject unresolved gyro-bias transients |
| gyro-bias tail standard deviation | `<= 0.00020 rad/s` | reject noisy gyro-bias estimates |
| accel-bias tail drift | `<= 0.05 m/s^2` | reject unresolved accel-bias transients |
| accel-bias tail standard deviation | `<= 0.035 m/s^2` | reject noisy accel-bias estimates |
| mount covariance | max sigma `<= 2 deg` | require internal covariance to agree with the tail test |
| attitude covariance | max sigma `<= 6 deg` | avoid saving priors during weak attitude observability |
| GNSS health | recent issue count `<= 12` and not stale | prevent bad/rejected GNSS from validating priors |

The public API exposes this as `health.stable`. Callers should not replicate the table outside the library; the table documents the current implementation so saved-prior behavior is auditable.

## Sleep Gap Behavior

An IMU timestamp gap above `0.05 s` clears sample-to-sample coupling. The first IMU after the gap anchors a new interval; the runtime does not strapdown-propagate across the missing time.

| Gap class | Duration | Behavior |
| --- | --- | --- |
| short sleep | `<= 15 min` | keep `Running`; navigation stays usable; apply bounded stationary covariance aging |
| medium sleep | `> 15 min` and `<= 1 h` | apply bounded stationary covariance aging; enter `DegradedDeadReckoning` if covariance remains usable |
| long sleep | `> 1 h` | enter `AwaitingGnssReseed` until an acceptable GNSS sample arrives |

During medium sleep, IMU prediction continues after wake in degraded dead-reckoning mode. If covariance grows beyond the navigation usability gate before GNSS returns, the state becomes `AwaitingGnssReseed`.

The degraded dead-reckoning gate is covariance-based:

| Block | Usability gate |
| --- | --- |
| horizontal position | `<= 30 m` sigma |
| horizontal velocity | `<= 2.5 m/s` sigma |
| roll/pitch attitude | `<= 5 deg` sigma |
| yaw attitude | `<= 15 deg` sigma |

When a GNSS sample is accepted in reseed mode, the runtime reseeds navigation from GNSS while preserving the last mount estimate and raw-body IMU bias priors. That is not a full reset to unknown mount. The preserved calibration covariance is never made more confident than either the previous covariance or these conservative floors:

| Preserved block | Minimum one-sigma floor after reseed |
| --- | --- |
| gyro bias | `0.03 deg/s` |
| accel bias | `0.05 m/s^2` |
| mount residual angle | `0.50 deg` |

This keeps calibration continuity useful without pretending that a GNSS reseed has re-observed the invariant states.

## Covariance Aging

Sleep aging adds uncertainty; it does not clamp the covariance to an absolute value. For a diagonal term, the operation is:

$$
\sigma_\text{new} = \sqrt{\sigma_\text{old}^2 + \sigma_\text{added}^2}.
$$

Short sleep adds up to these one-sigma terms at 15 minutes:

| State block | Added sigma at 15 min |
| --- | --- |
| horizontal position | `2 m` |
| vertical position | `1 m` |
| velocity | `0.25 m/s` |
| roll/pitch attitude | `0.25 deg` |
| yaw attitude | `1 deg` |
| gyro bias | `0.002 deg/s` |
| accel bias | `0.01 m/s^2` |
| mount | unchanged |

Medium sleep starts from the same short-sleep floor and grows to these one-sigma terms at one hour:

| State block | Added sigma range |
| --- | --- |
| horizontal position | `2 m` to `8 m` |
| vertical position | `1 m` to `4 m` |
| velocity | `0.25 m/s` to `0.75 m/s` |
| roll/pitch attitude | `0.25 deg` to `1 deg` |
| yaw attitude | `1 deg` to `5 deg` |
| gyro bias | `0.002 deg/s` to `0.010 deg/s` |
| accel bias | `0.01 m/s^2` to `0.03 m/s^2` |
| mount | unchanged |

Mount is not aged by sleep because the physical device mount should not change while the device is asleep. If the mount can change, the caller should not preserve the context.

## iOS Expectations

The iOS app follows the same lifecycle:

- normal Drive stop/start keeps the Rust `SensorFusion` context;
- replay/source changes reset fusion because the input stream is a different world;
- mount memory is applied only when starting a fresh context;
- mount memory should be saved only from stable health.

This means a user can pause and resume live collection without forcing a new alignment cycle. A true new recording source, however, should not inherit navigation state from the previous source.
