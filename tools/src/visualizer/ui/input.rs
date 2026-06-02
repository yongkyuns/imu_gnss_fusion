//! Shared visualizer input controls and synthetic scenario definitions.

use eframe::egui;
use serde::Deserialize;

#[cfg(not(target_arch = "wasm32"))]
pub(super) const DATASET_MANIFEST_PATH: &str = "web/datasets/manifest.json";

#[derive(Clone, Deserialize)]
pub(super) struct HostedDatasetEntry {
    #[serde(default)]
    pub(super) id: Option<String>,
    #[serde(default)]
    pub(super) label: Option<String>,
    #[serde(default)]
    pub(super) description: Option<String>,
    #[serde(default, alias = "baseUrl")]
    pub(super) base_url: Option<String>,
    #[serde(default, alias = "imu_csv")]
    pub(super) imu: Option<String>,
    #[serde(default, alias = "gnss_csv")]
    pub(super) gnss: Option<String>,
    #[serde(default, alias = "imu_csv_gz")]
    pub(super) imu_gz: Option<String>,
    #[serde(default, alias = "gnss_csv_gz")]
    pub(super) gnss_gz: Option<String>,
    #[serde(default, alias = "reference_attitude_csv")]
    pub(super) reference_attitude: Option<String>,
    #[serde(default, alias = "reference_attitude_csv_gz")]
    pub(super) reference_attitude_gz: Option<String>,
    #[serde(default, alias = "reference_mount_csv")]
    pub(super) reference_mount: Option<String>,
    #[serde(default, alias = "reference_mount_csv_gz")]
    pub(super) reference_mount_gz: Option<String>,
    #[serde(default, alias = "reference_position_csv")]
    pub(super) reference_position: Option<String>,
    #[serde(default, alias = "reference_position_csv_gz")]
    pub(super) reference_position_gz: Option<String>,
    #[serde(default, alias = "reference_motion_csv")]
    pub(super) reference_motion: Option<String>,
    #[serde(default, alias = "reference_motion_csv_gz")]
    pub(super) reference_motion_gz: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct HostedDatasetManifest {
    #[serde(default)]
    pub(super) datasets: Vec<HostedDatasetEntry>,
}

impl HostedDatasetEntry {
    pub(super) fn display_label(&self) -> String {
        self.label
            .as_deref()
            .or(self.id.as_deref())
            .unwrap_or("unnamed dataset")
            .to_string()
    }

    pub(super) fn picker_group_label(&self) -> &'static str {
        let id = self.id.as_deref().unwrap_or_default();
        let label = self.label.as_deref().unwrap_or_default();
        if id.starts_with("ios-") || label.starts_with("iOS ") {
            "iOS recordings"
        } else {
            "UBX/reference datasets"
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum InputMode {
    Synthetic,
    RealData,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NativeRealDataSource {
    CustomDirectory,
    ManifestDataset,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum SyntheticScenario {
    CityBlocks,
    FigureEight,
    FigureEightEarlyVelocityFault,
    FigureEightRollExcitation,
    StraightAccelBrake,
    ObservabilityStraight,
    ObservabilityAccelBrake,
    ObservabilityTurns,
    ObservabilityTurnsAccel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum SyntheticNoise {
    Truth,
    Low,
    Mid,
    High,
}

pub(super) const SYNTHETIC_SCENARIOS: &[SyntheticScenario] = &[
    SyntheticScenario::CityBlocks,
    SyntheticScenario::FigureEight,
    SyntheticScenario::FigureEightEarlyVelocityFault,
    SyntheticScenario::FigureEightRollExcitation,
    SyntheticScenario::StraightAccelBrake,
    SyntheticScenario::ObservabilityStraight,
    SyntheticScenario::ObservabilityAccelBrake,
    SyntheticScenario::ObservabilityTurns,
    SyntheticScenario::ObservabilityTurnsAccel,
];

pub(super) const SYNTHETIC_NOISE_PRESETS: &[SyntheticNoise] = &[
    SyntheticNoise::Truth,
    SyntheticNoise::Low,
    SyntheticNoise::Mid,
    SyntheticNoise::High,
];

impl SyntheticScenario {
    pub(super) fn display_label(self) -> &'static str {
        match self {
            Self::CityBlocks => "City blocks",
            Self::FigureEight => "Figure eight",
            Self::FigureEightEarlyVelocityFault => "Figure eight early GNSS fault",
            Self::FigureEightRollExcitation => "Figure eight roll excitation + GNSS fault",
            Self::StraightAccelBrake => "Straight accel/brake",
            Self::ObservabilityStraight => "Mount recovery: straight constant",
            Self::ObservabilityAccelBrake => "Mount recovery: accel/brake",
            Self::ObservabilityTurns => "Mount recovery: turns only",
            Self::ObservabilityTurnsAccel => "Mount recovery: turns + accel/brake",
        }
    }

    pub(super) fn scenario_text(self) -> (&'static str, &'static str) {
        match self {
            Self::CityBlocks => ("city_blocks_builtin.scenario", CITY_BLOCKS_SCENARIO),
            Self::FigureEight => ("figure8_builtin.scenario", FIGURE_EIGHT_SCENARIO),
            Self::FigureEightEarlyVelocityFault => (
                "figure8_early_gnss_fault_builtin.scenario",
                FIGURE_EIGHT_SCENARIO,
            ),
            Self::FigureEightRollExcitation => (
                "figure8_roll_excitation_builtin.scenario",
                FIGURE_EIGHT_ROLL_EXCITATION_SCENARIO,
            ),
            Self::StraightAccelBrake => (
                "straight_accel_brake_builtin.scenario",
                STRAIGHT_ACCEL_BRAKE_SCENARIO,
            ),
            Self::ObservabilityStraight => (
                "observability_straight_builtin.scenario",
                OBSERVABILITY_STRAIGHT_SCENARIO,
            ),
            Self::ObservabilityAccelBrake => (
                "observability_accel_brake_builtin.scenario",
                OBSERVABILITY_ACCEL_BRAKE_SCENARIO,
            ),
            Self::ObservabilityTurns => (
                "observability_turns_builtin.scenario",
                OBSERVABILITY_TURNS_SCENARIO,
            ),
            Self::ObservabilityTurnsAccel => (
                "observability_turns_accel_builtin.scenario",
                OBSERVABILITY_TURNS_ACCEL_SCENARIO,
            ),
        }
    }

    pub(super) fn early_fault(self) -> ([f64; 3], Option<[f64; 2]>) {
        match self {
            Self::FigureEightEarlyVelocityFault | Self::FigureEightRollExcitation => {
                ([0.5, 0.0, 0.0], Some([120.0, 360.0]))
            }
            Self::ObservabilityStraight
            | Self::ObservabilityAccelBrake
            | Self::ObservabilityTurns
            | Self::ObservabilityTurnsAccel => ([0.8, 0.25, 0.0], Some([130.0, 250.0])),
            Self::CityBlocks | Self::FigureEight | Self::StraightAccelBrake => {
                ([0.0, 0.0, 0.0], None)
            }
        }
    }

    pub(super) fn mount_rpy_deg(self) -> [f64; 3] {
        [5.0, -5.0, 5.0]
    }
}

impl SyntheticNoise {
    pub(super) fn display_label(self) -> &'static str {
        match self {
            Self::Truth => "None",
            Self::Low => "Low noise",
            Self::Mid => "Mid noise",
            Self::High => "High noise",
        }
    }

    pub(super) fn tooltip(self) -> &'static str {
        match self {
            Self::Truth => {
                "None\n\
                 IMU and GNSS are exact generated measurements.\n\
                 Use this to isolate filter formulation from sensor noise."
            }
            Self::Low => {
                "Low noise\n\
                 IMU: gyro ARW 0.05 deg/sqrt(hr), gyro bias drift 1 deg/hr\n\
                 IMU: accel VRW 0.015 m/s/sqrt(hr), accel bias drift 0.0002 m/s^2\n\
                 GNSS: position sigma 0.8 m horizontal, 1.2 m vertical\n\
                 GNSS: velocity sigma 0.03 m/s horizontal, 0.05 m/s vertical"
            }
            Self::Mid => {
                "Mid noise, consumer-grade reference point\n\
                 IMU: gyro ARW 0.3 deg/sqrt(hr), gyro bias drift 10 deg/hr\n\
                 IMU: accel VRW 0.05 m/s/sqrt(hr), accel bias drift 0.001 m/s^2\n\
                 GNSS: position sigma 3 m horizontal, 5 m vertical\n\
                 GNSS: velocity sigma 0.10 m/s horizontal, 0.15 m/s vertical"
            }
            Self::High => {
                "High noise\n\
                 IMU: gyro ARW 1.0 deg/sqrt(hr), gyro bias drift 30 deg/hr\n\
                 IMU: accel VRW 0.12 m/s/sqrt(hr), accel bias drift 0.005 m/s^2\n\
                 GNSS: position sigma 8 m horizontal, 12 m vertical\n\
                 GNSS: velocity sigma 0.30 m/s horizontal, 0.50 m/s vertical"
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn cli_value(self) -> &'static str {
        match self {
            Self::Truth => "truth",
            Self::Low => "low",
            Self::Mid => "mid",
            Self::High => "high",
        }
    }
}

impl From<SyntheticNoise> for crate::visualizer::pipeline::synthetic::SyntheticNoiseMode {
    fn from(value: SyntheticNoise) -> Self {
        match value {
            SyntheticNoise::Truth => Self::Truth,
            SyntheticNoise::Low => Self::Low,
            SyntheticNoise::Mid => Self::Mid,
            SyntheticNoise::High => Self::High,
        }
    }
}

impl From<crate::visualizer::pipeline::synthetic::SyntheticNoiseMode> for SyntheticNoise {
    fn from(value: crate::visualizer::pipeline::synthetic::SyntheticNoiseMode) -> Self {
        match value {
            crate::visualizer::pipeline::synthetic::SyntheticNoiseMode::Truth => Self::Truth,
            crate::visualizer::pipeline::synthetic::SyntheticNoiseMode::Low => Self::Low,
            crate::visualizer::pipeline::synthetic::SyntheticNoiseMode::Mid => Self::Mid,
            crate::visualizer::pipeline::synthetic::SyntheticNoiseMode::High => Self::High,
        }
    }
}

pub(super) fn draw_run_button(
    ui: &mut egui::Ui,
    enabled: bool,
    busy: bool,
    progress: f32,
    text: &str,
) -> bool {
    let width = 128.0_f32.max(ui.spacing().interact_size.x * 3.2);
    let height = ui.spacing().interact_size.y;
    if busy {
        ui.add_sized(
            [width, height],
            egui::ProgressBar::new(progress.clamp(0.0, 1.0))
                .desired_width(width)
                .desired_height(height)
                .fill(ui.visuals().selection.bg_fill)
                .text(egui::RichText::new(format!(
                    "{text} {:>3.0}%",
                    100.0 * progress.clamp(0.0, 1.0)
                ))),
        );
        false
    } else {
        ui.add_enabled(
            enabled,
            egui::Button::new("Run").min_size(egui::vec2(width, height)),
        )
        .clicked()
    }
}

pub(super) fn draw_synthetic_noise_help(ui: &mut egui::Ui) {
    ui.label(egui::RichText::new("Synthetic noise presets").strong());
    ui.add_space(2.0);
    ui.label(
        "Noise is applied to generated IMU and GNSS measurements. Values are 1-sigma figures.",
    );
    ui.add_space(8.0);

    egui::Grid::new("synthetic_noise_help_grid")
        .num_columns(5)
        .spacing([16.0, 5.0])
        .striped(true)
        .show(ui, |ui| {
            ui.strong("Preset");
            ui.strong("Gyro");
            ui.strong("Accel");
            ui.strong("GNSS pos");
            ui.strong("GNSS vel");
            ui.end_row();

            noise_help_row(ui, "None", "exact", "exact", "exact", "exact");
            noise_help_row(
                ui,
                "Low noise",
                "0.05 deg/sqrt(hr)\n1 deg/hr drift",
                "0.015 m/s/sqrt(hr)\n0.0002 m/s^2 drift",
                "0.8 m horiz\n1.2 m vert",
                "0.03 m/s horiz\n0.05 m/s vert",
            );
            noise_help_row(
                ui,
                "Mid noise",
                "0.3 deg/sqrt(hr)\n10 deg/hr drift",
                "0.05 m/s/sqrt(hr)\n0.001 m/s^2 drift",
                "3 m horiz\n5 m vert",
                "0.10 m/s horiz\n0.15 m/s vert",
            );
            noise_help_row(
                ui,
                "High noise",
                "1.0 deg/sqrt(hr)\n30 deg/hr drift",
                "0.12 m/s/sqrt(hr)\n0.005 m/s^2 drift",
                "8 m horiz\n12 m vert",
                "0.30 m/s horiz\n0.50 m/s vert",
            );
        });
}

fn noise_help_row(
    ui: &mut egui::Ui,
    preset: &str,
    gyro: &str,
    accel: &str,
    gnss_pos: &str,
    gnss_vel: &str,
) {
    ui.label(preset);
    ui.label(gyro);
    ui.label(accel);
    ui.label(gnss_pos);
    ui.label(gnss_vel);
    ui.end_row();
}

const CITY_BLOCKS_SCENARIO: &str = r#"
initial lat=32 lon=120 alt=0 speed=0 yaw=0 pitch=0 roll=0
wait 20s
repeat 3 {
    accelerate 1.0m/s^2 for 8s
    wait 10s
    turn left 10dps for 9s
    wait 10s
    brake 1.0m/s^2 for 8s
    wait 10s
    accelerate 1.0m/s^2 for 8s
    wait 10s
    turn right 10dps for 9s
    wait 10s
    brake 1.0m/s^2 for 8s
    wait 10s
}
"#;

const FIGURE_EIGHT_SCENARIO: &str = r#"
initial lat=32 lon=120 alt=0 speed=0 yaw=0 pitch=0 roll=0
wait 60s
accelerate 0.6m/s^2 for 20s
wait 10s
repeat 11 {
    turn left 10dps for 36s
    turn right 10dps for 36s
}
brake 0.6666667m/s^2 for 18s
"#;

const FIGURE_EIGHT_ROLL_EXCITATION_SCENARIO: &str = r#"
initial lat=32 lon=120 alt=0 speed=0 yaw=0 pitch=0 roll=0
wait 60s
accelerate 0.6m/s^2 for 20s
wait 10s
repeat 11 {
    drive yaw=10 roll=0.25 for=18s
    drive yaw=10 roll=-0.25 for=18s
    drive yaw=-10 roll=-0.25 for=18s
    drive yaw=-10 roll=0.25 for=18s
}
brake 0.6666667m/s^2 for 18s
"#;

const STRAIGHT_ACCEL_BRAKE_SCENARIO: &str = r#"
initial lat=32 lon=120 alt=0 speed=0 yaw=0 pitch=0 roll=0
wait 20s
repeat 2 {
    accelerate 0.5m/s^2 for 20s
    wait 20s
    brake 0.5m/s^2 for 20s
    wait 15s
}
"#;

const OBSERVABILITY_STRAIGHT_SCENARIO: &str = r#"
initial lat=32 lon=120 alt=0 speed=0 yaw=0 pitch=0 roll=0
wait 20s
accelerate 0.8 for=20s
turn left 10 for=18s
turn right 10 for=18s
coast for=24s
wait for=420s
"#;

const OBSERVABILITY_ACCEL_BRAKE_SCENARIO: &str = r#"
initial lat=32 lon=120 alt=0 speed=0 yaw=0 pitch=0 roll=0
wait 20s
accelerate 0.8 for=20s
turn left 10 for=18s
turn right 10 for=18s
coast for=24s
repeat 12 {
    accelerate 1.0 for=15s
    coast for=10s
    brake 1.0 for=15s
    coast for=10s
}
"#;

const OBSERVABILITY_TURNS_SCENARIO: &str = r#"
initial lat=32 lon=120 alt=0 speed=0 yaw=0 pitch=0 roll=0
wait 20s
accelerate 0.8 for=20s
turn left 10 for=18s
turn right 10 for=18s
coast for=24s
repeat 10 {
    turn left 10 for=24s
    turn right 10 for=24s
}
"#;

const OBSERVABILITY_TURNS_ACCEL_SCENARIO: &str = r#"
initial lat=32 lon=120 alt=0 speed=0 yaw=0 pitch=0 roll=0
wait 20s
accelerate 0.8 for=20s
turn left 10 for=18s
turn right 10 for=18s
coast for=24s
repeat 8 {
    accelerate 0.8 for=10s
    turn left 10 for=20s
    brake 0.8 for=10s
    turn right 10 for=20s
}
"#;
