use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Deserialize)]
struct ReplayOutput {
    time_s: Vec<f64>,
    pos_ecef_m: Vec<[f64; 3]>,
    vel_ecef_mps: Vec<[f64; 3]>,
    q_es: Vec<[f64; 4]>,
    q_cs: Vec<[f64; 4]>,
}

#[derive(Debug, Deserialize)]
struct DiagRow {
    time_s: f64,
    applied_obs_types: Vec<i32>,
}

#[derive(Debug, Deserialize)]
struct GoldenSummary {
    sample_count: usize,
    checkpoints: Vec<GoldenCheckpoint>,
    applied_observations: Vec<GoldenObsRow>,
}

#[derive(Debug, Deserialize)]
struct GoldenCheckpoint {
    index: usize,
    time_s: f64,
    pos_ecef_m: [f64; 3],
    vel_ecef_mps: [f64; 3],
    q_es: [f64; 4],
    q_cs: [f64; 4],
}

#[derive(Debug, Deserialize)]
struct GoldenObsRow {
    time_s: f64,
    applied_obs_types: Vec<i32>,
}

#[test]
fn loose_short_fixture_matches_golden_observation_sequence_and_checkpoints() {
    let fixture_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/loose_nsr_short");
    let scratch_dir = std::env::temp_dir().join(format!("loose_parity_{}", std::process::id()));
    let _ = fs::remove_dir_all(&scratch_dir);
    fs::create_dir_all(&scratch_dir).expect("create scratch dir");
    let out_json = scratch_dir.join("out.json");
    let diag_json = scratch_dir.join("diag.json");

    let status = Command::new(run_loose_nsr_path())
        .arg("--input-dir")
        .arg(&fixture_dir)
        .arg("--init-json")
        .arg(fixture_dir.join("init.json"))
        .arg("--out-json")
        .arg(&out_json)
        .arg("--diag-json")
        .arg(&diag_json)
        .arg("--diag-until-s")
        .arg("25")
        .status()
        .expect("run_loose_nsr");
    assert!(
        status.success(),
        "run_loose_nsr failed with status {status}"
    );

    let output: ReplayOutput = load_json(&out_json);
    let diag: Vec<DiagRow> = load_json(&diag_json);
    let golden: GoldenSummary = load_json(&fixture_dir.join("golden_summary.json"));

    assert_eq!(output.time_s.len(), golden.sample_count, "sample count");
    assert_eq!(
        diag.len(),
        golden.applied_observations.len(),
        "diag row count"
    );

    for (actual, expected) in diag.iter().zip(&golden.applied_observations) {
        assert!(
            (actual.time_s - expected.time_s).abs() <= 1.0e-9,
            "diag time mismatch: actual={} expected={}",
            actual.time_s,
            expected.time_s
        );
        assert_eq!(
            actual.applied_obs_types, expected.applied_obs_types,
            "diag obs types mismatch at time {}",
            expected.time_s
        );
    }

    for checkpoint in &golden.checkpoints {
        let i = checkpoint.index;
        assert!(
            i < output.time_s.len(),
            "checkpoint index {} out of range {}",
            i,
            output.time_s.len()
        );
        assert!(
            (output.time_s[i] - checkpoint.time_s).abs() <= 1.0e-9,
            "checkpoint time mismatch at index {}: actual={} expected={}",
            i,
            output.time_s[i],
            checkpoint.time_s
        );
        assert!(
            vec_norm_diff3(output.pos_ecef_m[i], checkpoint.pos_ecef_m) <= 1.0e-2,
            "pos_ecef mismatch at index {}",
            i
        );
        assert!(
            vec_norm_diff3(output.vel_ecef_mps[i], checkpoint.vel_ecef_mps) <= 1.0e-4,
            "vel_ecef mismatch at index {}",
            i
        );
        assert!(
            quat_angle_deg(output.q_es[i], checkpoint.q_es) <= 1.0e-3,
            "q_es mismatch at index {}",
            i
        );
        assert!(
            quat_angle_deg(output.q_cs[i], checkpoint.q_cs) <= 1.0e-3,
            "q_cs mismatch at index {}",
            i
        );
    }

    let _ = fs::remove_dir_all(&scratch_dir);
}

fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> T {
    serde_json::from_slice(
        &fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display())),
    )
    .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn run_loose_nsr_path() -> PathBuf {
    let exe_dir = std::env::current_exe()
        .expect("current_exe")
        .parent()
        .expect("test exe parent")
        .parent()
        .expect("target dir")
        .to_path_buf();
    let mut candidate = exe_dir.join("run_loose_nsr");
    if cfg!(windows) {
        candidate.set_extension("exe");
    }
    assert!(
        candidate.is_file(),
        "run_loose_nsr binary not found at {}",
        candidate.display()
    );
    candidate
}

fn vec_norm_diff3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn quat_angle_deg(a: [f64; 4], b: [f64; 4]) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let na = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]).sqrt();
    let nb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2] + b[3] * b[3]).sqrt();
    let c = (dot.abs() / (na * nb)).clamp(-1.0, 1.0);
    2.0 * c.acos().to_degrees()
}
