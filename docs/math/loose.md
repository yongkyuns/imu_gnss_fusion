# Loose INS/GNSS Notes

The full loose filter formulation is now PDF-first:

- [Loose INS/GNSS formulation PDF](../loose_formulation.pdf)
- [Loose INS/GNSS formulation TEX](../loose_formulation.tex)

Operational references:

- Symbolic source: `ekf/ins_gnss_loose.py`
- Generated wrapper: `ekf/src/generated_loose.rs`
- Runtime/reference implementation: `ekf/src/loose.rs`

Regenerate generated loose snippets with:

```bash
python ekf/ins_gnss_loose.py --emit-rust
```

Useful focused checks after loose-model changes:

```bash
cargo test -p sensor-fusion --test loose_mount_observability --locked
cargo test -p sim --test loose_parity --locked
```
