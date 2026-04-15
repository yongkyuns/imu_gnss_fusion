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

## ESKF mount / heading status

Current target log:

- `logger/data/ubx_raw_20260328_153757.bin`

Current focus:

- Explain why seeded ESKF on `153757` converges to the wrong mount/vehicle-heading basin.
- Distinguish the first bad state change from later mount compensation.

Recent relevant work:

- Added a transition analyzer in:
  - `sim/src/bin/analyze_eskf_transition.rs`
- Added several temporary ESKF-side experiments in:
  - `ekf/c/src/sensor_fusion.c`
  - `ekf/c/src/sf_eskf.c`
  - `ekf/c/include/sensor_fusion_defs.h`
  - `ekf/c/include/sensor_fusion_internal.h`
  - `ekf/c/include/sf_eskf.h`

Important current conclusions:

- The wrong settled mount is **not** primarily caused by mount flexibility.
- The bad frame appears before the first strong mount-moving update.
- `BODY_VEL_Z` is **not** the first trigger.
  - Disabling only `BODY_VEL_Z` mount injection did not fix the bad frame.
  - Disabling `BODY_VEL_Z` entirely also did not fix the bad frame.
- `BODY_VEL_Y` is the first strong corrective update that visibly slams the state, but it is reacting to a frame that is already wrong.
- GNSS velocity is a major contributor, but not the sole cause.
  - Disabling only `GPS_VEL_D` helps the vertical channel somewhat, but large lateral body velocity remains.
  - Disabling all GNSS velocity reduces the bad frame substantially, but the frame still drifts wrong before the first strong body-velocity correction.
- Therefore the current root bucket is upstream of mount adaptation:
  - seeded predict-frame / handoff composition
  - plus GNSS-velocity coupling into nav state
  - then body-velocity updates react to the already-bad frame

Key analyzer findings on `153757`:

- Baseline:
  - `72.013 s`: `pre_body=[8.083, -2.272, 0.237]`, `pre_qcs=[1.00, -2.42, 8.06]`
  - `76.013 s`: `pre_body=[8.849, -5.268, -0.935]`, `pre_qcs=[1.00, -2.42, 8.06]`
  - `78.515 s`: `pre_body=[8.640, -5.139, -1.795]`, `pre_qcs=[1.00, -2.42, 8.06]`
  - First strong mount-moving step:
    - `78.525 s` `BODY_VEL_Z`
    - `pre_body=[8.645, -5.116, -1.793]`
    - `post_qcs=[0.45, -2.98, 8.73]`
- With `--disable-gps-vel-d`:
  - vertical channel improves somewhat
  - large lateral body velocity remains
- With `--disable-gps-vel-all`:
  - `72.013 s`: `pre_body=[7.901, -0.809, 0.103]`
  - `76.013 s`: `pre_body=[8.840, -3.684, -0.171]`
  - `78.515 s`: `pre_body=[8.756, -4.097, -0.760]`
  - so GNSS velocity matters, but the frame is still wrong before mount moves
- With `SF_ESKF_DIAG_DISABLE_BODY_VEL_Z=1` and `--disable-gps-vel-all`:
  - `72.013 s`: `pre_body=[7.774, -0.812, 1.052]`, `pre_qcs=[1.78, 1.57, 3.04]`
  - `76.013 s`: `pre_body=[8.849, -3.973, -0.147]`, `pre_qcs=[1.78, 1.57, 3.04]`
  - `78.515 s`: `pre_body=[8.775, -4.116, -0.901]`, `pre_qcs=[1.78, 1.57, 3.04]`
  - First strong update becomes:
    - `78.525 s` `BODY_VEL_Y`
    - `innov=4.104`
    - `pre_att=[-6.34, -3.86, 57.35] -> post_att=[-5.80, -5.88, 52.12]`
    - `pre_body=[8.785, -4.099, -0.906] -> post_body=[8.952, -2.465, -1.233]`
    - `pre_qcs=[1.78, 1.57, 3.04] -> post_qcs=[2.17, 1.55, 3.27]`

What this means:

- The filter is already in the wrong nav/body frame before the first strong body-velocity correction.
- Later mount motion is mostly compensation, not the original cause.
- Tuning mount stiffness after handoff does not fix the root cause because the leader state is nav/body-frame corruption, not mount drift.

Current best next step:

1. Inspect the seeded predict-frame/handoff composition directly:
   - nominal `q`
   - `eskf_mount_q_vb`
   - live align `mount_q_vb`
   - the implied body-frame velocity at and just after handoff
2. Verify whether yaw-only attitude initialization and seeded IMU pre-rotation are mutually consistent for the first few seconds after handoff.
3. Keep using `analyze_eskf_transition.rs` as the primary diagnostic tool for `72–79 s`.

Useful resume prompt:

- "Resume ESKF `153757` mount/heading debugging, read `agent.md`, use `analyze_eskf_transition.rs`, and continue from the finding that the bad frame exists before the first strong `BODY_VEL_Y/Z` mount update. Focus on seeded handoff and predict-frame consistency."

Next Codex session to start:

- Repo: `/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion`
- Branch: current working branch in that repo
- Entry doc: `agent.md`
- Suggested start prompt:
  - "Resume ESKF `153757` mount/heading debugging in `/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion`. Read `agent.md` first. Use `sim/src/bin/analyze_eskf_transition.rs` and continue from the finding that the bad frame exists before the first strong `BODY_VEL_Y/Z` mount update. Focus on seeded handoff and predict-frame consistency."
