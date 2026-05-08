# Full EKF Notes

The Full EKF formulation is now PDF-first:

- [Full EKF formulation PDF](../full.pdf)
- [Full EKF formulation TEX](../full.tex)

Operational references:

- Symbolic source: `sensor_fusion/src/full/formulation.py`
- Generated wrapper: `sensor_fusion/src/full/generated.rs`
- Runtime/reference implementation: `sensor_fusion/src/full/`

Regenerate generated Full EKF snippets with:

```bash
python sensor_fusion/src/full/formulation.py --emit-rust
```

Useful focused checks after Full EKF model changes:

```bash
cargo test -p sensor_fusion --test full_mount_observability --locked
cargo test -p sim --test full_parity --locked
```
