# EKF Algorithm Reference

The runtime filter is an error-state EKF with generated symbolic Jacobians and a
small embedded-oriented update path. This page is the map for the self-contained
algorithm reference.

## Core Pages

- [](frames.md): active rotations, frame names, mount convention, and injection sides.
- [](runtime-ekf.md): nominal state, error state, propagation, generated
  Jacobians, measurement rows, scalar/batch update algebra, GNSS gating, NHC
  scheduling, vehicle-roll prior, injection, reset, and initialization.
- [](ekf-matrices.md): state/noise ordering, complete sparse transition and
  noise-input matrix structure, generated measurement Jacobians, and update
  forms.
- [](mount-states.md): why mount is estimated inside the EKF, why align is
  separate, how mount/attitude ambiguity arises, and why full navigation differs
  from AHRS-only attitude filters.
- [](align.md): reduced mount-alignment estimator used before automatic EKF
  initialization.
- [](observability.md): practical mount observability claims and
  limits.
- [](roll-observability.md): derivation appendix for NHC, covariance coupling,
  and roll ambiguity.

## Generated-Code Notes

Generated-code workflow details live in [](../development/generated-models.md).
The public algorithm reference describes the equations and runtime behavior; the
developer page describes how the checked-in generated Rust fragments are updated.
