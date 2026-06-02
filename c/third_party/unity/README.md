# Unity Test Shim

This directory contains a small Unity-compatible C test shim used by the C99
sensor-fusion scaffold. It implements only the assertions needed by the current
repo-local tests and keeps the C backend build self-contained.

If the project later vendors the upstream ThrowTheSwitch Unity package, keep the
include path and `RUN_TEST`/`TEST_ASSERT_*` surface compatible so `c/Makefile`
and tests do not need to change.
