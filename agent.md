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
