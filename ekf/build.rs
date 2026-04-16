use std::env;
use std::path::{Path, PathBuf};

fn first_existing_path(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|path| path.exists()).cloned()
}

fn wasi_sysroot() -> Option<PathBuf> {
    if let Ok(path) = env::var("WASI_SYSROOT") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }

    if let Ok(path) = env::var("WASI_SDK_PATH") {
        let sysroot = Path::new(&path).join("share/wasi-sysroot");
        if sysroot.exists() {
            return Some(sysroot);
        }
    }

    first_existing_path(&[
        PathBuf::from("/usr/local/opt/wasi-libc/share/wasi-sysroot"),
        PathBuf::from("/opt/homebrew/opt/wasi-libc/share/wasi-sysroot"),
    ])
}

fn llvm_tool(name: &str) -> Option<PathBuf> {
    if let Ok(root) = env::var("LLVM_ROOT") {
        let tool = Path::new(&root).join("bin").join(name);
        if tool.exists() {
            return Some(tool);
        }
    }

    first_existing_path(&[
        PathBuf::from(format!("/usr/local/opt/llvm/bin/{name}")),
        PathBuf::from(format!("/opt/homebrew/opt/llvm/bin/{name}")),
    ])
}

fn main() {
    println!("cargo:rerun-if-env-changed=SF_ESKF_BODY_VEL_USE_QCS_CONJ");
    println!("cargo:rerun-if-env-changed=SF_ESKF_DIAG_DISABLE_BODY_VEL_Y_MOUNT");
    println!("cargo:rerun-if-env-changed=SF_ESKF_DIAG_DISABLE_GPS_VEL_D");
    println!("cargo:rerun-if-env-changed=SF_ESKF_DIAG_DISABLE_BODY_VEL_Z");
    println!("cargo:rerun-if-env-changed=SF_ESKF_DIAG_DISABLE_BODY_VEL_Z_MOUNT");
    println!("cargo:rerun-if-changed=c/Makefile");
    println!("cargo:rerun-if-changed=c/generated");
    println!("cargo:rerun-if-changed=c/generated_eskf");
    println!("cargo:rerun-if-changed=c/generated_loose");
    println!("cargo:rerun-if-changed=c/include");
    println!("cargo:rerun-if-changed=c/src");

    let target = env::var("TARGET").unwrap_or_default();
    let mut build = cc::Build::new();
    build
        .std("c11")
        .opt_level(3)
        .define("NDEBUG", None)
        .include("c")
        .include("c/include")
        .file("c/src/sensor_fusion.c")
        .file("c/src/sf_align.c")
        .file("c/src/sf_stationary_mount.c")
        .file("c/src/sf_eskf.c")
        .file("c/src/sf_loose.c");

    if target == "wasm32-wasip1" {
        if let Some(clang) = llvm_tool("clang") {
            build.compiler(clang);
        }
        if let Some(llvm_ar) = llvm_tool("llvm-ar") {
            build.archiver(llvm_ar);
        }
        if let Some(sysroot) = wasi_sysroot() {
            build.flag("--target=wasm32-wasip1");
            build.flag(&format!("--sysroot={}", sysroot.display()));
        } else {
            println!(
                "cargo:warning=WASI sysroot not found; set WASI_SYSROOT or WASI_SDK_PATH for wasm32-wasip1 builds"
            );
        }
    }

    if env::var("SF_ESKF_BODY_VEL_USE_QCS_CONJ")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        build.define("SF_ESKF_BODY_VEL_USE_QCS_CONJ", Some("1"));
    }
    if env::var("SF_ESKF_DIAG_DISABLE_BODY_VEL_Y_MOUNT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        build.define("SF_ESKF_DIAG_DISABLE_BODY_VEL_Y_MOUNT", Some("1"));
    }
    if env::var("SF_ESKF_DIAG_DISABLE_GPS_VEL_D")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        build.define("SF_ESKF_DIAG_DISABLE_GPS_VEL_D", Some("1"));
    }
    if env::var("SF_ESKF_DIAG_DISABLE_BODY_VEL_Z")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        build.define("SF_ESKF_DIAG_DISABLE_BODY_VEL_Z", Some("1"));
    }
    if env::var("SF_ESKF_DIAG_DISABLE_BODY_VEL_Z_MOUNT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        build.define("SF_ESKF_DIAG_DISABLE_BODY_VEL_Z_MOUNT", Some("1"));
    }
    build.compile("sensor_fusion_c_impl");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=m");
    }
}
