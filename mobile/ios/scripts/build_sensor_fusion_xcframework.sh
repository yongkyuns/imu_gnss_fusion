#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IOS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CRATE_DIR="${SENSOR_FUSION_FFI_CRATE_DIR:-$IOS_DIR/SensorFusionFFI}"
MANIFEST_PATH="$CRATE_DIR/Cargo.toml"
HEADER_PATH="$CRATE_DIR/include/sensor_fusion_ffi.h"
LIB_NAME="${SENSOR_FUSION_FFI_LIB_NAME:-sensor_fusion_ffi}"
BUILD_DIR="${SENSOR_FUSION_FFI_BUILD_DIR:-$IOS_DIR/build}"
RUST_TARGET_DIR="$BUILD_DIR/rust-target"
PACKAGE_DIR="$BUILD_DIR/xcframework-input"
OUTPUT_XCFRAMEWORK="$BUILD_DIR/SensorFusionFFI.xcframework"
BUILD_PROFILE="${SENSOR_FUSION_FFI_PROFILE:-release}"

REQUIRED_TARGETS=(
  "aarch64-apple-ios"
  "aarch64-apple-ios-sim"
)

OPTIONAL_TARGETS=(
  "x86_64-apple-ios"
)

fail() {
  echo "error: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command '$1'"
}

is_target_installed() {
  local target="$1"
  rustup target list --installed | grep -Fxq "$target"
}

profile_dir_name() {
  if [[ "$BUILD_PROFILE" == "release" ]]; then
    echo "release"
  else
    echo "$BUILD_PROFILE"
  fi
}

rust_build_args() {
  if [[ "$BUILD_PROFILE" == "release" ]]; then
    echo "--release"
  else
    echo "--profile=$BUILD_PROFILE"
  fi
}

find_static_library() {
  local target="$1"
  local profile_dir="$2"
  local target_output_dir="$RUST_TARGET_DIR/$target/$profile_dir"
  local expected="$target_output_dir/lib$LIB_NAME.a"

  if [[ -f "$expected" ]]; then
    echo "$expected"
    return
  fi

  local matches=()
  while IFS= read -r match; do
    matches+=("$match")
  done < <(find "$target_output_dir" -maxdepth 1 -type f -name 'lib*.a' | sort)

  if [[ "${#matches[@]}" -eq 1 ]]; then
    echo "${matches[0]}"
    return
  fi

  if [[ "${#matches[@]}" -eq 0 ]]; then
    fail "no Rust static library found for $target in $target_output_dir. Expected lib$LIB_NAME.a; set SENSOR_FUSION_FFI_LIB_NAME if the crate uses another staticlib name."
  fi

  fail "multiple Rust static libraries found for $target in $target_output_dir. Set SENSOR_FUSION_FFI_LIB_NAME to choose the correct library."
}

require_command cargo
require_command rustup
require_command lipo
require_command xcodebuild

[[ -f "$MANIFEST_PATH" ]] || fail "missing FFI crate manifest at $MANIFEST_PATH"
[[ -f "$HEADER_PATH" ]] || fail "missing FFI header at $HEADER_PATH"

missing_targets=()
for target in "${REQUIRED_TARGETS[@]}"; do
  if ! is_target_installed "$target"; then
    missing_targets+=("$target")
  fi
done

if [[ "${#missing_targets[@]}" -gt 0 ]]; then
  fail "missing required Rust iOS target(s): ${missing_targets[*]}. Install them with: rustup target add ${missing_targets[*]}"
fi

build_targets=("${REQUIRED_TARGETS[@]}")
for target in "${OPTIONAL_TARGETS[@]}"; do
  if is_target_installed "$target"; then
    build_targets+=("$target")
  else
    echo "info: skipping optional Rust target $target; install with 'rustup target add $target' to include it"
  fi
done

mkdir -p "$BUILD_DIR" "$PACKAGE_DIR"
rm -rf "$PACKAGE_DIR" "$OUTPUT_XCFRAMEWORK" "$BUILD_DIR/include"
mkdir -p "$PACKAGE_DIR" "$BUILD_DIR/include"
cp "$HEADER_PATH" "$BUILD_DIR/include/sensor_fusion_ffi.h"

profile_dir="$(profile_dir_name)"
read -r -a cargo_profile_args <<<"$(rust_build_args)"

device_lib=""
sim_libs=()
for target in "${build_targets[@]}"; do
  echo "building SensorFusionFFI for $target"
  CARGO_TARGET_DIR="$RUST_TARGET_DIR" cargo build \
    --manifest-path "$MANIFEST_PATH" \
    --target "$target" \
    "${cargo_profile_args[@]}"

  lib_path="$(find_static_library "$target" "$profile_dir")"
  case "$target" in
    *-apple-ios-sim | x86_64-apple-ios)
      sim_libs+=("$lib_path")
      ;;
    *)
      device_lib="$lib_path"
      ;;
  esac
done

xcframework_args=()

if [[ -n "$device_lib" ]]; then
  slice_dir="$PACKAGE_DIR/ios-device"
  headers_dir="$slice_dir/Headers"
  mkdir -p "$headers_dir"
  cp "$HEADER_PATH" "$headers_dir/sensor_fusion_ffi.h"
  cp "$device_lib" "$slice_dir/lib$LIB_NAME.a"

  xcframework_args+=(
    -library "$slice_dir/lib$LIB_NAME.a"
    -headers "$headers_dir"
  )
fi

if [[ "${#sim_libs[@]}" -gt 0 ]]; then
  slice_dir="$PACKAGE_DIR/ios-simulator"
  headers_dir="$slice_dir/Headers"
  mkdir -p "$headers_dir"
  cp "$HEADER_PATH" "$headers_dir/sensor_fusion_ffi.h"
  if [[ "${#sim_libs[@]}" -eq 1 ]]; then
    cp "${sim_libs[0]}" "$slice_dir/lib$LIB_NAME.a"
  else
    lipo -create "${sim_libs[@]}" -output "$slice_dir/lib$LIB_NAME.a"
  fi

  xcframework_args+=(
    -library "$slice_dir/lib$LIB_NAME.a"
    -headers "$headers_dir"
  )
fi

if [[ "${#xcframework_args[@]}" -eq 0 ]]; then
  fail "no libraries were built for XCFramework packaging"
fi

echo "creating $OUTPUT_XCFRAMEWORK"
xcodebuild -create-xcframework "${xcframework_args[@]}" -output "$OUTPUT_XCFRAMEWORK"

echo "wrote $OUTPUT_XCFRAMEWORK"
echo "wrote $BUILD_DIR/include/sensor_fusion_ffi.h"
