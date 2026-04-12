# Agent Notes

## Loose filter interpretation

Keep this straight when debugging loose:

- The propagated attitude state is `q_es`.
- In the current visualizer/runtime loose setup, IMU can be intentionally pre-rotated by a frozen coarse mount seed before predict.
- With that setup, loose predict is propagating the seeded IMU frame to ECEF.
- `qcs` is a residual chassis/vehicle alignment state used in measurement models like NHC.
- `qcs` not appearing in mechanization is not automatically a bug.

Do not regress on this assumption:

- "predict must use `qcs`" is not the default diagnosis here.

Debug priorities when loose mount blows up:

1. Check the GNSS velocity update path and covariance use.
2. Check frame consistency of the fed velocity/position measurements.
3. Check interaction between frozen coarse seed, residual `qcs`, and NHC updates.
4. Only then revisit whether the state formulation itself is wrong.

## WIP Resume Note

Current branch:

- `loose-matlab-parity`

Latest relevant commits:

- `4abe6ed` `Add seeded loose stress simulation harness`
- `ff50612` `Debug seeded loose replay behavior`
- `772497f` `Stabilize seeded loose mount handling`
- `8000565` `Document loose filter frame assumptions`

Current target log:

- `logger/data/ubx_raw_20260328_153757.bin`

Recent relevant changes:

- Added analyzer/visualizer diagnostics fixes in `ff50612`.
- Added a standalone synthetic seeded loose stress harness in:
  - `sim/run_seeded_loose_stress_sim.py`
  - `sim/motion_profiles/seeded_highspeed_straight_10min.csv`
- The stress harness generates a synthetic `gnss-ins-sim` dataset, initializes the loose C filter in the same seeded pre-rotated-IMU mode as the real visualizer, and sweeps:
  - coarse seed mount error
  - early GNSS down-velocity bias windows
  - GNSS velocity over-trust
  - injected accelerometer z-channel bias

Important current conclusions:

- The earlier `~ -6 deg` loose vehicle pitch was partly a diagnostics/display artifact.
- After fixing the displayed car-attitude composition, the real remaining car pitch on `153757` is closer to `~ -2.5 to -3 deg`, not `-6 deg`.
- Early `NHC` is the main driver of the bad mount branch on the real log.
  - At the first big event around `226.55 s`, disabling `NHC` removes the mount jump.
  - Disabling GNSS velocity alone does not remove it; GNSS velocity amplifies it but is not sufficient by itself.
- Delaying `NHC` is not a real fix.
  - It postpones the first bad branch, but the filter later converges to the same wrong mount solution.
- The filter continues to update attitude while nearly stationary because there is no stationary-aware mode / ZUPT / low-speed gating for loose.
  - Near-standstill roll changes are mostly `q_es` continuing to absorb GNSS-velocity + NHC residuals.
- On synthetic self-consistent cases, the loose C core behaves well and matches Matlab/Python closely enough for the relevant debugging.
  - This points away from a gross loose-core mechanization bug.
- The new seeded synthetic stress harness reproduces the wrong-mount / wrong-pitch basin with practical flat-road motion when an early GNSS down-velocity inconsistency is injected.
  - `seed_only` is benign.
  - `vd_bias_only` is enough to drive the filter into a wrong mount/pitch basin.
  - Combining seed error with early `v_d` bias makes the wrong basin more pronounced.
- The synthetic stress harness does **not** reproduce the real strong negative `accel_bias.z` drift.
  - Injecting constant accelerometer z-bias only produces relatively small negative estimated `b_a_z`.
  - Therefore the real `b_a_z` issue likely involves a second mechanism beyond the wrong-mount / wrong-pitch basin.

Current best next step:

1. Focus on the real-log vertical channel, not on `qcs`-in-predict.
2. Investigate whether the remaining real `accel_bias.z` drift comes from replay-specific GNSS vertical-velocity modeling/trust rather than the same mechanism that causes the wrong mount basin.
3. Most promising next synthetic probe:
   - anisotropic GNSS velocity covariance, especially weakening only `v_d`
   - then compare against the real `153757` bias behavior
4. If debugging real replay directly:
   - inspect how `v_d` and its covariance behave around loose init and later stationary windows
   - compare low-speed handling and vertical-velocity trust against the synthetic seeded stress results

Useful resume prompt:

- "Resume loose filter debugging on `loose-matlab-parity`, read `agent.md`, and continue diagnosing the remaining `153757` seeded loose vertical-channel / accel-z bias issue. Use the seeded synthetic stress harness as the controlled baseline for wrong-mount / wrong-pitch behavior."
