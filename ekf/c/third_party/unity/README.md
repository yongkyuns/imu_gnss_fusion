Unity should be vendored into:

- `third_party/unity/src/unity.c`
- `third_party/unity/src/unity.h`
- `third_party/unity/src/unity_internals.h`

The `Makefile` expects that layout.

This repository intentionally does not auto-download Unity during the build.
The C library and tests must remain offline-buildable and reproducible.
