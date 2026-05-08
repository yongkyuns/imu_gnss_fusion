# Full EKF Notes

The Full EKF formulation is now PDF-first:

- [Full EKF formulation PDF](../full_formulation.pdf)
- [Full EKF formulation TEX](../full_formulation.tex)

Operational references:

- Symbolic source: `sensor_fusion/ins_gnss_full.py`
- Generated wrapper: `sensor_fusion/src/generated_full.rs`
- Runtime/reference implementation: `sensor_fusion/src/full.rs`

Regenerate generated Full EKF snippets with:

```bash
python sensor_fusion/ins_gnss_full.py --emit-rust
```

Useful focused checks after Full EKF model changes:

```bash
cargo test -p sensor_fusion --test full_mount_observability --locked
cargo test -p sim --test full_parity --locked
```
