use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("wasm32") {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let repo_dir = manifest_dir
        .parent()
        .expect("tools crate must live below repo root");
    let c_dir = repo_dir.join("c");
    let c_build_dir = c_dir.join("build");

    for path in [
        "Makefile",
        "sensor_fusion/include/sensor_fusion.h",
        "sensor_fusion/src/align.c",
        "sensor_fusion/src/align.h",
        "sensor_fusion/src/sensor_fusion.c",
        "sensor_fusion/src/ekf/runtime.c",
        "sensor_fusion/src/ekf/runtime.h",
        "sensor_fusion/src/ekf/generated_model.c",
        "sensor_fusion/src/ekf/generated_model.h",
        "road_events/include/road_events.h",
        "road_events/src/road_events.c",
    ] {
        println!("cargo:rerun-if-changed={}", c_dir.join(path).display());
    }

    let status = Command::new("make")
        .arg("-C")
        .arg(&c_dir)
        .arg("lib")
        .status()
        .expect("failed to run make for C backend");
    if !status.success() {
        panic!("C backend archive build failed");
    }

    println!("cargo:rustc-link-search=native={}", c_build_dir.display());
    println!("cargo:rustc-link-lib=static=imu_gnss_fusion_c");
    if target.contains("linux")
        || target.contains("darwin")
        || target.contains("freebsd")
        || target.contains("netbsd")
        || target.contains("openbsd")
    {
        println!("cargo:rustc-link-lib=m");
    }
}
