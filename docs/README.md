# Documentation

This directory collects project-level notes for the IMU/GNSS fusion workspace. The root [README](../README.md) is the entrypoint for setup and common workflows; these pages keep deeper math, testing, and architecture details out of the main guide.

## Start Here

- [Testing](testing.md): local test commands, targeted suites, fixtures, and expensive-data notes.
- [Frame conventions](math/frames.md): short index for navigation, ECEF, body, vehicle, seeded, and corrected frames.
- [Loose INS/GNSS notes](math/loose.md): concise operational links for the loose filter.
- [Simulation tooling map](../sim/README.md): stable `sim` binaries, generic replay schema, and supported visualizer modes.
- [Browser visualizer](../web/README.md): wasm build and static hosting instructions.

## Detailed Math Notes

- [ESKF mount formulation PDF](eskf_mount_formulation.pdf) and [TEX](eskf_mount_formulation.tex).
- [Align/NHC formulation PDF](align_nhc_formulation.pdf) and [TEX](align_nhc_formulation.tex).
- [Loose INS/GNSS formulation PDF](loose_formulation.pdf) and [TEX](loose_formulation.tex).
## Documentation Conventions

- Keep README-level material short and operational.
- Prefer `sim/README.md` for crate-specific tool inventory and diagnostics.
- Put estimator conventions and equations in the formulation PDFs under `docs/`, with TEX kept as the editable source.
- Keep `docs/math/*.md` as index and operational-reference pages, not duplicate derivations.
- When generated Rust changes, update both the symbolic source notes and the testing notes if the verification path changes.
