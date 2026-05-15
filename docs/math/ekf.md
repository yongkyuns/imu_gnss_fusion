# EKF Math Notes

The detailed EKF derivations are PDF-first. Editable TEX sources live beside
the PDFs under `docs/`. Keep `docs/math/*.md` pages as short operational
indexes rather than duplicate derivations.

Regenerate generated EKF snippets with:

```bash
python sensor_fusion/src/ekf/formulation.py --emit-rust
```

Useful focused checks after model changes:

```bash
cargo test -p sensor_fusion --locked
cargo test -p sim --test synthetic_gnss_ins --locked
```
