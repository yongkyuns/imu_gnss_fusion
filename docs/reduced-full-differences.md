# Reduced vs Full Investigation Notes

This note records known Reduced/Full differences that can affect convergence
comparisons. It is intentionally diagnostic: these are not all bugs, but they
must be accounted for before treating a Reduced/Full behavior difference as a
mathematical formulation issue.

## Current Figure-Eight Finding

Dataset: `web/datasets/parking-lot-figure8-nominal-001`.

Both filters receive the same align seed at `27.342 s`. The first post-seed
sample has the same mount yaw error (`+1.184 deg`) and mount yaw sigma
(`6.000 deg`) for Reduced and Full, so the later divergence is not caused by
different initial mount quaternions.

Final reference-relative mount errors from the replay:

| Filter | Mount qerr | Roll err | Pitch err | Yaw err |
| --- | ---: | ---: | ---: | ---: |
| Align | `1.299 deg` | `-0.194 deg` | `+0.151 deg` | `-1.277 deg` |
| Reduced | `2.333 deg` | `-0.334 deg` | `+2.245 deg` | `-0.546 deg` |
| Full | `9.595 deg` | `-0.253 deg` | `+2.402 deg` | `-9.294 deg` |

The large Full mount quaternion error is therefore almost entirely mount yaw.
Full also carries larger vehicle attitude error, especially yaw, so this is a
vehicle-yaw/mount-yaw allocation problem rather than a pure mount-only metric
artifact.

## Confirmed Differences

### Initial Vehicle Yaw Covariance

Reduced has separate initial attitude covariance for roll/pitch and yaw:

- roll/pitch attitude sigma: `attitude_roll_pitch_init_sigma_deg`
- yaw attitude sigma: `yaw_init_sigma_deg`

Full now mirrors this split:

- roll/pitch attitude sigma: `full::InitConfig::attitude_sigma_deg`
- yaw attitude sigma: `full::InitConfig::attitude_yaw_sigma_deg`

Before this split, Full used one attitude sigma for all three axes. Tightening
Full roll/pitch attitude covariance to improve roll/mount behavior also
tightened vehicle yaw, making Full overconfident in yaw during the early
figure-eight convergence window.

Focused Full-only sweep on the figure-eight replay, with mount yaw sigma fixed
at `6 deg`:

| Full vehicle yaw sigma | Final Full mount yaw error | Final Full mount qerr |
| ---: | ---: | ---: |
| `2 deg` | `-9.296 deg` | `9.597 deg` |
| `4 deg` | `-4.787 deg` | `5.615 deg` |
| `6 deg` | `-3.009 deg` | `4.274 deg` |
| `8 deg` | `-2.090 deg` | `3.665 deg` |

This is the strongest confirmed cause of the observed Full/Reduced gap on this
dataset.

After adding the separate Full yaw covariance field and setting the default
Full yaw sigma to `6 deg`, the standard `first_divergence` replay improved
Full from mount qerr `9.595 deg` / yaw error `-9.294 deg` to mount qerr
`8.058 deg` / yaw error `-7.505 deg`. The remaining gap indicates that yaw
covariance was a real contributor but not the only difference.

### Full Covariance Basis At Initialization

Full stores position, velocity, and attitude errors in ECEF-oriented bases, but
the initialization uncertainty values are specified in local NED/vehicle terms.
Writing the local roll/pitch/yaw covariance directly onto the Full ECEF
attitude diagonal distorted the intended covariance. On the parking-lot
figure-eight replay, the first post-init local attitude sigmas were roughly
`[4.46, 2.28, 4.41] deg` instead of the intended `[2, 2, 6] deg`.

Full initialization now rotates the local covariance blocks into the Full error
basis:

- position and velocity: `P_e = C_en P_n C_en^T`
- attitude: `P_e_theta = C_ev P_v_theta C_ev^T`

This is a formulation/initialization fix, not a tuning knob. After the basis
fix, the same replay improved Full from mount qerr `8.058 deg` / yaw error
`-7.505 deg` to mount qerr `4.212 deg` / yaw error `-3.070 deg`. Reduced stayed
at mount qerr `2.333 deg` / yaw error `-0.546 deg`.

### GNSS Scheduling

Reduced stores pending GNSS samples and fuses them at the next IMU epoch. If the
GNSS is stale, Reduced can still fuse GNSS without same-epoch NHC.

Full only fuses the latest GNSS sample when the next IMU sample is within the
freshness window currently used by the facade. If the age window is missed, that
GNSS row can be skipped by Full. This can remove early corrective information
from Full while Reduced still receives it.

### Initialization Heading Source

Reduced uses `gnss.heading_rad` when it is available, falling back to GNSS
velocity course only when needed.

Full currently initializes vehicle yaw from GNSS velocity course and does not
use the optional GNSS heading field. On low-speed or transient maneuvers this
can give Full a different vehicle-yaw prior even with the same mount seed.

### Correction Injection And Reset

Reduced injects attitude and mount corrections using a first-order small-angle
quaternion. Its reset path applies the reset Jacobian to the attitude block.

Full injects attitude and mount corrections using finite Euler-composition
quaternions and applies reset blocks to both attitude and mount covariance. This
can change the covariance lifecycle after repeated cross-covariance-driven
updates and should be tested separately from measurement tuning.

### Mount Observability Path

GNSS and NHC/body-velocity measurement rows do not directly observe mount in
either filter. Mount corrections arrive through propagated cross-covariance with
velocity and attitude states. Because of this, covariance initialization,
propagation, reset, and scheduling can dominate how a bad mount/vehicle yaw
allocation recovers.

## Confirmation Tests To Run

1. Verify Full with `attitude_sigma_deg = 2 deg` and
   `attitude_yaw_sigma_deg = 6 deg` on the parking-lot figure-eight replay.
2. Make Full use the same GNSS heading initialization rule as Reduced, then
   rerun the same replay.
3. Count skipped Full GNSS rows during the first minute and compare against
   Reduced pending-GNSS fusion.
4. Test correction injection/reset parity separately from tuning by changing one
   mechanism at a time and rerunning the same replay plus synthetic clean
   figure-eight cases.

## Remaining Figure-Eight Gap After Full Yaw-Covariance Split

After separating Full roll/pitch attitude covariance from Full yaw attitude
covariance, the parking-lot figure-eight replay still shows a significant Full
mount-yaw gap:

| Metric | Reduced | Full |
| --- | ---: | ---: |
| Final mount qerr | `2.333 deg` | `8.058 deg` |
| Final mount yaw error | `-0.546 deg` | `-7.505 deg` |
| Final mount yaw sigma | `1.660 deg` | `1.407 deg` |

Additional checks narrowed the remaining root cause:

- **Heading source is not the cause for this dataset.** At the initialization
  GNSS sample (`27.342 s`), `heading_rad` and `atan2(ve, vn)` are identical
  (`34.507689 deg`). They also match through the high-speed startup window.
- **GNSS age scheduling is not the cause for this dataset.** All `246`
  post-initialization GNSS rows are eligible for Full's `0..=0.05 s` age gate.
  The next-IMU age is about `0.005 s` on average and below `0.010 s` in the
  relevant windows.
- **Full measurement gating is a secondary contributor.** Full accepted both
  GNSS position and velocity at `236` of `246` post-init epochs; one epoch
  accepted neither position nor velocity even though it was age-eligible. This
  is not a scheduling drop, but it can reduce corrective information.

The dominant remaining difference is update allocation:

| Window | GNSS mount-yaw correction Full-Reduced | NHC mount-yaw correction Full-Reduced |
| --- | ---: | ---: |
| `28-40 s` | `-0.268 deg` | `+3.658 deg` |
| `40-70 s` | `-0.463 deg` | `+3.066 deg` |
| `70-100 s` | `+1.462 deg` | `-0.625 deg` |
| `100-150.843 s` | `+0.883 deg` | `-1.111 deg` |

Totals over `28.0-150.843 s`:

- Reduced NHC mount-yaw correction: `-15.571 deg`.
- Full NHC mount-yaw correction: `-10.583 deg`.
- NHC Full-minus-Reduced correction deficit: `+4.988 deg`.
- Reduced GNSS mount-yaw correction: `-3.092 deg`.
- Full GNSS mount-yaw correction: `-1.478 deg`.
- GNSS Full-minus-Reduced correction deficit: `+1.614 deg`.

Thus roughly `76%` of the remaining mount-yaw correction deficit comes from NHC
allocation, not GNSS. Full NHC is active, but it preserves a different
attitude-yaw/mount-yaw split:

| Time | Reduced corr(att yaw, mount yaw) | Full corr(att yaw, mount yaw) |
| ---: | ---: | ---: |
| `40 s` | `-0.804` | `-0.506` |
| `70 s` | `-0.905` | `-0.740` |
| `150 s` | `-0.872` | `-0.757` |

This points to covariance lifecycle and cross-covariance allocation rather than
a missing NHC measurement.

### Mount Reset Diagnostic

Full applies a reset block to mount covariance after mount injection, while
Reduced resets the attitude block only. A temporary diagnostic run skipped only
the Full mount reset block while preserving nominal mount injection and
attitude reset.

Result on the same parking-lot figure-eight replay:

| Full mode | Final mount qerr | Roll err | Pitch err | Yaw err |
| --- | ---: | ---: | ---: | ---: |
| Default reset | `8.058 deg` | `-0.475 deg` | `+2.936 deg` | `-7.505 deg` |
| Skip mount reset | `7.973 deg` | `-0.286 deg` | `+2.383 deg` | `-7.611 deg` |

Skipping the mount reset barely changes the outcome and slightly worsens mount
yaw. Therefore the Full mount-reset block is not the primary cause of the
remaining yaw gap on this dataset.

The next root-cause target is the effective NHC covariance allocation itself:
both filters rely on covariance cross-correlation for mount updates, and the
Full NHC rows still place less net correction into mount yaw even after heading
source, GNSS age scheduling, and mount reset are ruled out for this log.

### NHC Allocation After Diagnostic Cadence Parity

The `covariance_history` diagnostic previously drove its standalone Full path
with NHC at IMU cadence while the public `SensorFusion` facade decimates NHC to
the configured period. The diagnostic was corrected to use the same period-based
Full NHC schedule as runtime. This changes diagnostic innovation statistics but
does not change filter behavior.

With cadence parity and the corrected Full covariance basis over `28-150 s`:

| NHC row | Reduced count | Full count | Reduced net mount yaw | Full net mount yaw |
| --- | ---: | ---: | ---: | ---: |
| Y | `211` | `250` | `-15.580 deg` | `-14.736 deg` |
| Z | `211` | `250` | `+0.009 deg` | `+0.194 deg` |

Full NHC-Y residuals are larger, not smaller:

| NHC row | Reduced mean abs innovation | Full mean abs innovation |
| --- | ---: | ---: |
| Y | `0.753 m/s` | `0.748 m/s` |
| Z | `0.047 m/s` | `0.051 m/s` |

The remaining weaker Full correction comes from lower effective mount gain,
not lack of residual. The NHC-Y mount-gain norm accumulated by the diagnostic
is about `0.00151` for Reduced versus `0.00104` for Full. This aligns with the
covariance snapshots: by `150 s`, `corr(att_yaw, mount_yaw)` is about `-0.872`
for Reduced but only `-0.786` for Full, and Full's mount-yaw sigma is lower
(`1.53 deg` versus `1.66 deg`) despite the larger yaw error.

Scale and bias absorption were checked separately and are not the dominant
sink: over the same interval Full and Reduced bias corrections are similar,
while Full is missing several degrees of mount-yaw correction. Near-freezing
Full scale covariance did not recover the missing mount correction.

### Current Root-Cause State

Two concrete Full covariance-frame issues have been fixed:

- Full initial position, velocity, and attitude covariance is now rotated from
  local NED/vehicle axes into the ECEF error-state basis instead of being placed
  on ECEF diagonals directly.
- Full GNSS position/velocity whitening now rotates NED measurement covariance
  with `C_ne^T R_n C_ne`, matching the ECEF state basis.

On the parking-lot figure-eight replay, the first fix is material: Full final
mount quaternion error dropped from about `8.06 deg` to about `4.21 deg`.
The GNSS measurement-covariance rotation fix is mathematically required but is
not the dominant remaining effect on this dataset; Full remains around
`4.30 deg` final mount quaternion error versus Reduced around `2.33 deg`.

The remaining error now has a clearer sequence:

| Interval | Main difference | Evidence |
| --- | --- | --- |
| `30.9-35.4 s` | NHC-Y starts a different attitude-yaw/mount-yaw split | Full-minus-Reduced NHC-Y mount-yaw correction is about `-1.32 deg`, while attitude-yaw correction differs by about `+3.95 deg`. |
| `40.0-42.9 s` | GNSS velocity becomes the largest net mount-yaw allocation difference | Full-minus-Reduced GNSS-velocity mount-yaw correction is about `+2.09 deg`; Full attitude error grows at the same time. |
| `28.0-150.8 s` | Full applies less net negative mount-yaw correction overall | Full-minus-Reduced mount-yaw correction is about `+2.74 deg`, mostly from GNSS velocity (`+1.70 deg`) and NHC-Y (`+0.38 deg`). |

At the final row (`150.734 s`), Reduced mount error is roughly
`[-0.334, +2.245, -0.546] deg`, while Full is roughly
`[-0.221, +2.730, -3.321] deg`. Full also carries lower mount-yaw covariance
(`1.52 deg` sigma versus Reduced `1.66 deg`) despite the larger yaw error,
which means the remaining issue is not a missing residual. It is a covariance
allocation / nominal-state divergence problem: Full becomes more confident in
the wrong yaw split and then GNSS velocity/NHC updates have less ability to
move the mount state back.

A direct-NED Full GNSS row experiment was also tested. It replaced the
Cholesky-whitened ECEF rows with local-NED scalar rows. The result was neutral
to slightly worse (`~4.34 deg` final Full mount quaternion error), so row
whitening basis alone does not explain the remaining gap.

### Row-Level GNSS Parity Diagnostic

`covariance_history` now has a diagnostic-only per-row CSV writer:

```bash
cargo run --release -p sim --bin covariance_history -- \
  --generic-replay-dir /tmp/fig8_generic \
  --times 41.5 \
  --summary-window 40,43 \
  --gnss-parity-csv /tmp/fig8_gnss_parity.csv \
  --gnss-parity-window 40,43 \
  --max-time-s 44
```

The CSV records scalar GNSS rows for both filters with signed residual,
effective residual, innovation variance, NIS, inferred `K` into attitude/mount,
and per-row `dx` transformed into the Reduced-style common basis. This was
added because aggregate `vel_xy` diagnostics hide which individual row creates
the Full/Reduced split.

On the parking-lot figure-eight `40-43 s` window, the row-level CSV has the
expected `72` rows (`6` GNSS epochs, `6` rows, `2` filters). It confirms that
the large Full event is not a row-count issue:

| System/rows | Count | Effective residual abs | NIS sum | Net mount-yaw dx |
| --- | ---: | ---: | ---: | ---: |
| Reduced velocity N/E/D | `18` | `4.315` | `13.046` | `+1.974 deg` |
| Full velocity X/Y/Z | `18` | `13.031` | `11.702` | `+4.067 deg` |

The dominant Full rows in this interval are whitened velocity `Y/Z`:

| Full row | Count | Effective residual abs | Net attitude-yaw dx | Net mount-yaw dx |
| --- | ---: | ---: | ---: | ---: |
| `vel_y` | `6` | `5.119` | `-1.692 deg` | `+1.556 deg` |
| `vel_z` | `6` | `5.349` | `-1.772 deg` | `+2.516 deg` |

The row-level view sharpens the current diagnosis: by `40-43 s`, Full is
entering the same accepted GNSS epochs with a different nominal yaw/mount split
and larger velocity residuals. The next experiment should therefore isolate the
earlier `30.9-35.4 s` NHC-Y covariance allocation and the Full mount covariance
reset/cross-covariance lifecycle, rather than changing GNSS row scheduling.
