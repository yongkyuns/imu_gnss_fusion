# Documentation

This directory collects project-level notes for the IMU/GNSS fusion workspace. The root [README](../README.md) is the entrypoint for setup and common workflows; these pages keep deeper math, testing, and architecture details out of the main guide.

## Start Here

- [Testing](testing.md): local test commands, targeted suites, fixtures, and expensive-data notes.
- [Frame conventions](math/frames.md): navigation, body, vehicle, seeded, and corrected frames used across align, loose, and ESKF code.
- [Loose INS/GNSS notes](math/loose.md): state layout, update sources, generated-code path, and diagnostic conventions for the loose filter.
- [Simulation tooling map](../sim/README.md): stable `sim` binaries, generic replay schema, and supported visualizer modes.

## Detailed Math Notes

- [ESKF mount formulation PDF](eskf_mount_formulation.pdf) and [TEX](eskf_mount_formulation.tex).
- [Align/NHC formulation PDF](align_nhc_formulation.pdf) and [TEX](align_nhc_formulation.tex).
- [Align pitch observability note PDF](align_pitch_observability_note.pdf) and [TEX](align_pitch_observability_note.tex).

## Architecture Assets

- [Repository architecture diagram](repo_architecture.png).
- [Penpot source for the architecture diagram](repo_architecture.pen).

## Documentation Conventions

- Keep README-level material short and operational.
- Prefer `sim/README.md` for crate-specific tool inventory and diagnostics.
- Put estimator conventions and equations under `docs/math/`.
- When generated Rust changes, update both the symbolic source notes and the testing notes if the verification path changes.
