use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=c/Makefile");
    println!("cargo:rerun-if-changed=c/generated");
    println!("cargo:rerun-if-changed=c/generated_eskf");
    println!("cargo:rerun-if-changed=c/generated_loose");
    println!("cargo:rerun-if-changed=c/include");
    println!("cargo:rerun-if-changed=c/src");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    build_archive(
        &out_dir,
        "sensor_fusion_c_impl",
        &[
            PathBuf::from("c/src/sensor_fusion.c"),
            PathBuf::from("c/src/sf_align.c"),
            PathBuf::from("c/src/sf_stationary_mount.c"),
            PathBuf::from("c/src/sf_eskf.c"),
            PathBuf::from("c/src/sf_loose.c"),
        ],
        &["-Ic", "-Ic/include"],
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=sensor_fusion_c_impl");
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=m");
}

fn build_archive(out_dir: &PathBuf, lib_name: &str, sources: &[PathBuf], includes: &[&str]) {
    let mut objs = Vec::with_capacity(sources.len());
    for (idx, src) in sources.iter().enumerate() {
        let obj = out_dir.join(format!("{lib_name}_{idx}.o"));
        let mut cmd = Command::new("cc");
        cmd.arg("-std=c11").arg("-O3").arg("-DNDEBUG");
        for include in includes {
            cmd.arg(include);
        }
        cmd.arg("-c").arg(src).arg("-o").arg(&obj);
        let status = cmd.status().expect("failed to run C compiler");
        assert!(
            status.success(),
            "C compilation failed for {}",
            src.display()
        );
        objs.push(obj);
    }

    let lib = out_dir.join(format!("lib{lib_name}.a"));
    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&lib);
    for obj in &objs {
        ar.arg(obj);
    }
    let status = ar.status().expect("failed to run ar");
    assert!(
        status.success(),
        "archive creation failed for {}",
        lib.display()
    );
}
