#!/usr/bin/env python3
import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: run_with_timeout.py <timeout-seconds> <command...>", file=sys.stderr)
        return 2

    timeout_s = float(sys.argv[1])
    cmd = sys.argv[2:]

    try:
        proc = subprocess.run(cmd, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"timed out after {timeout_s:.1f}s: {' '.join(cmd)}", file=sys.stderr)
        return 124
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
