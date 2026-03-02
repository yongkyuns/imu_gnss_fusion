use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=c/ekf.c");
    println!("cargo:rerun-if-changed=c/ekf.h");
    println!("cargo:rerun-if-changed=c/generated");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let obj = out_dir.join("ekf.o");
    let lib = out_dir.join("libekf_c_impl.a");

    let cc_status = Command::new("cc")
        .arg("-std=c11")
        .arg("-O3")
        .arg("-DNDEBUG")
        .arg("-Ic")
        .arg("-c")
        .arg("c/ekf.c")
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("failed to run C compiler");
    assert!(cc_status.success(), "C compilation failed");

    let ar_status = Command::new("ar")
        .arg("crus")
        .arg(&lib)
        .arg(&obj)
        .status()
        .expect("failed to run ar");
    assert!(ar_status.success(), "archive creation failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=ekf_c_impl");
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=m");
}
