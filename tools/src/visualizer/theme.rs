use eframe::egui::{
    self, Color32, CornerRadius, FontFamily, FontId, Stroke, Style, TextStyle, Theme, Visuals,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiDensity {
    Comfortable,
    Compact,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UiTheme {
    Light,
    #[default]
    Dark,
}

impl UiTheme {
    pub fn from_value(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "light" | "positron" => Some(Self::Light),
            "dark" | "dark-matter" | "dark_matter" => Some(Self::Dark),
            _ => None,
        }
    }

    pub fn display_label(self) -> &'static str {
        match self {
            Self::Light => "Light",
            Self::Dark => "Dark",
        }
    }

    pub fn storage_value(self) -> &'static str {
        match self {
            Self::Light => "light",
            Self::Dark => "dark",
        }
    }

    fn egui_theme(self) -> Theme {
        match self {
            Self::Light => Theme::Light,
            Self::Dark => Theme::Dark,
        }
    }
}

pub fn apply(ctx: &egui::Context, density: UiDensity, theme: UiTheme) {
    ctx.set_theme(theme.egui_theme());
    ctx.set_style(build_style(density, theme));
    ctx.set_visuals(build_visuals(theme));
}

fn build_style(density: UiDensity, theme: UiTheme) -> Style {
    let mut style = Style::default();
    let scale = match density {
        UiDensity::Comfortable => 1.0,
        UiDensity::Compact => 0.82,
    };

    style.text_styles = [
        (
            TextStyle::Heading,
            FontId::new(22.0 * scale, FontFamily::Proportional),
        ),
        (
            TextStyle::Name("Heading2".into()),
            FontId::new(20.0 * scale, FontFamily::Proportional),
        ),
        (
            TextStyle::Body,
            FontId::new(15.5 * scale, FontFamily::Proportional),
        ),
        (
            TextStyle::Monospace,
            FontId::new(14.0 * scale, FontFamily::Monospace),
        ),
        (
            TextStyle::Button,
            FontId::new(15.0 * scale, FontFamily::Proportional),
        ),
        (
            TextStyle::Small,
            FontId::new(13.0 * scale, FontFamily::Proportional),
        ),
    ]
    .into();

    style.spacing.item_spacing = egui::vec2(10.0 * scale, 10.0 * scale);
    style.spacing.button_padding = egui::vec2(12.0 * scale, 8.0 * scale);
    style.spacing.menu_margin = egui::Margin::same((10.0 * scale).round() as i8);
    style.spacing.window_margin = egui::Margin::same((8.0 * scale).round() as i8);
    style.spacing.indent = 18.0 * scale;
    style.spacing.combo_width = 148.0 * scale;
    style.spacing.slider_width = 180.0 * scale;
    style.spacing.interact_size = egui::vec2(42.0 * scale, 28.0 * scale);
    style.visuals = build_visuals(theme);

    style
}

fn build_visuals(theme: UiTheme) -> Visuals {
    let accent = Color32::from_rgb(230, 122, 76);
    let accent_soft = Color32::from_rgb(198, 97, 59);
    let (mut visuals, panel, panel_alt, panel_strong, text, text_muted, border, code_bg) =
        match theme {
            UiTheme::Dark => (
                Visuals::dark(),
                Color32::from_rgb(17, 21, 28),
                Color32::from_rgb(24, 29, 38),
                Color32::from_rgb(31, 37, 48),
                Color32::from_rgb(233, 238, 245),
                Color32::from_rgb(153, 165, 180),
                Color32::from_rgb(58, 69, 84),
                Color32::from_rgb(20, 25, 33),
            ),
            UiTheme::Light => (
                Visuals::light(),
                Color32::from_rgb(247, 244, 240),
                Color32::from_rgb(239, 234, 227),
                Color32::from_rgb(229, 222, 213),
                Color32::from_rgb(43, 49, 59),
                Color32::from_rgb(111, 120, 132),
                Color32::from_rgb(191, 178, 165),
                Color32::from_rgb(243, 237, 230),
            ),
        };

    visuals.override_text_color = Some(text);
    visuals.panel_fill = panel;
    visuals.window_fill = panel_alt;
    visuals.faint_bg_color = panel_alt;
    visuals.extreme_bg_color = panel_strong;
    visuals.code_bg_color = code_bg;
    visuals.warn_fg_color = Color32::from_rgb(255, 196, 107);
    visuals.error_fg_color = Color32::from_rgb(255, 108, 108);
    visuals.hyperlink_color = Color32::from_rgb(120, 188, 255);
    visuals.selection.bg_fill = accent;
    visuals.selection.stroke = Stroke::new(
        1.0,
        match theme {
            UiTheme::Dark => Color32::from_rgb(255, 221, 209),
            UiTheme::Light => Color32::from_rgb(117, 61, 39),
        },
    );

    visuals.window_corner_radius = CornerRadius::same(16);
    visuals.menu_corner_radius = CornerRadius::same(12);
    visuals.window_stroke = Stroke::new(1.0, border);
    visuals.widgets.noninteractive.bg_fill = panel_alt;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, border);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, text_muted);

    visuals.widgets.inactive.bg_fill = panel_strong;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, border);
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, text);

    visuals.widgets.hovered.bg_fill = match theme {
        UiTheme::Dark => Color32::from_rgb(42, 50, 64),
        UiTheme::Light => Color32::from_rgb(235, 226, 216),
    };
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, accent);
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, text);
    visuals.widgets.hovered.expansion = 0.0;

    visuals.widgets.active.bg_fill = accent;
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, accent);
    visuals.widgets.active.fg_stroke = Stroke::new(
        1.0,
        match theme {
            UiTheme::Dark => Color32::WHITE,
            UiTheme::Light => text,
        },
    );

    visuals.widgets.open.bg_fill = panel_strong;
    visuals.widgets.open.bg_stroke = Stroke::new(1.0, accent_soft);
    visuals.widgets.open.fg_stroke = Stroke::new(1.0, text);

    visuals
}
