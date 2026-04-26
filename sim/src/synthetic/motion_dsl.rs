use anyhow::{Result, bail};

use super::gnss_ins_path::{InitialState, MotionCommand, MotionProfile};

pub fn parse_motion_dsl(text: &str) -> Result<MotionProfile> {
    let lines = preprocess(text);
    let mut initial = None;
    let mut idx = 0usize;
    let mut commands = Vec::new();
    while idx < lines.len() {
        let line = lines[idx].as_str();
        let keyword = first_token(line);
        match keyword {
            Some("initial") | Some("init") => {
                initial = Some(parse_initial(line)?);
                idx += 1;
            }
            Some("repeat") => {
                let (repeated, next_idx) = parse_repeat(&lines, idx)?;
                commands.extend(repeated);
                idx = next_idx;
            }
            Some("{") => bail!("unexpected '{{' at line {}", idx + 1),
            Some("}") => bail!("unexpected '}}' at line {}", idx + 1),
            Some(_) => {
                commands.push(parse_command(line)?);
                idx += 1;
            }
            None => idx += 1,
        }
    }
    let Some(initial) = initial else {
        bail!("motion DSL must include an initial line");
    };
    if commands.is_empty() {
        bail!("motion DSL must include at least one command");
    }
    Ok(MotionProfile { initial, commands })
}

fn preprocess(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let line = line
                .split('#')
                .next()
                .unwrap_or("")
                .split("//")
                .next()
                .unwrap_or("")
                .trim();
            if line.is_empty() {
                None
            } else {
                Some(line.to_string())
            }
        })
        .collect()
}

fn parse_initial(line: &str) -> Result<InitialState> {
    let fields = KeyValues::from_line(line);
    Ok(InitialState {
        lat_deg: fields.number(&["lat", "latitude"])?.unwrap_or(32.0),
        lon_deg: fields.number(&["lon", "longitude"])?.unwrap_or(120.0),
        height_m: fields.number(&["alt", "height", "h"])?.unwrap_or(0.0),
        vel_body_mps: [
            fields
                .number(&["vx", "forward_speed", "speed"])?
                .unwrap_or(0.0),
            fields.number(&["vy", "lateral_speed"])?.unwrap_or(0.0),
            fields.number(&["vz", "vertical_speed"])?.unwrap_or(0.0),
        ],
        yaw_pitch_roll_deg: [
            fields.number(&["yaw"])?.unwrap_or(0.0),
            fields.number(&["pitch"])?.unwrap_or(0.0),
            fields.number(&["roll"])?.unwrap_or(0.0),
        ],
    })
}

fn parse_block(lines: &[String], mut idx: usize) -> Result<(Vec<MotionCommand>, usize)> {
    let mut commands = Vec::new();
    while idx < lines.len() {
        let line = lines[idx].as_str();
        match first_token(line) {
            Some("}") => return Ok((commands, idx + 1)),
            Some("initial") | Some("init") => bail!("initial line is only valid at top level"),
            Some("repeat") => {
                let (repeated, next_idx) = parse_repeat(lines, idx)?;
                commands.extend(repeated);
                idx = next_idx;
            }
            Some("{") => bail!("unexpected '{{' at line {}", idx + 1),
            Some(_) => {
                commands.push(parse_command(line)?);
                idx += 1;
            }
            None => idx += 1,
        }
    }
    bail!("unterminated repeat block")
}

fn parse_repeat(lines: &[String], idx: usize) -> Result<(Vec<MotionCommand>, usize)> {
    let tokens = split_tokens(&lines[idx]);
    if tokens.len() < 2 {
        bail!("repeat requires a count");
    }
    let count = tokens[1]
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("invalid repeat count '{}'", tokens[1]))?;
    let mut block_start = idx + 1;
    if !tokens.iter().any(|token| *token == "{") {
        if lines.get(block_start).map(|line| line.trim()) != Some("{") {
            bail!("repeat block must start with '{{'");
        }
        block_start += 1;
    }
    let (block, next_idx) = parse_block(lines, block_start)?;
    let mut commands = Vec::with_capacity(block.len() * count);
    for _ in 0..count {
        commands.extend(block.iter().copied());
    }
    Ok((commands, next_idx))
}

fn parse_command(line: &str) -> Result<MotionCommand> {
    let tokens = split_tokens(line);
    if tokens.is_empty() {
        bail!("empty motion command");
    }
    let fields = KeyValues::from_tokens(&tokens);
    let gps_visible = parse_gps_visibility(&tokens, &fields)?;
    let duration_s = parse_duration(&tokens, &fields)?;
    let mut yaw_pitch_roll_cmd_deg = [0.0; 3];
    let mut body_cmd = [0.0; 3];

    match tokens[0] {
        "wait" | "hold" | "coast" => {}
        "accelerate" | "accel" => {
            body_cmd[0] = parse_positional_number(&tokens, 1, "accelerate")?;
        }
        "brake" | "decelerate" => {
            body_cmd[0] = -parse_positional_number(&tokens, 1, "brake")?.abs();
        }
        "turn" => {
            let direction = tokens
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("turn requires left/right or a signed rate"))?;
            let (sign, rate_idx) = match *direction {
                "left" | "l" => (1.0, 2),
                "right" | "r" => (-1.0, 2),
                _ => (1.0, 1),
            };
            yaw_pitch_roll_cmd_deg[0] =
                sign * parse_positional_number(&tokens, rate_idx, "turn")?.abs();
        }
        "yaw" => {
            yaw_pitch_roll_cmd_deg[0] = parse_positional_number(&tokens, 1, "yaw")?;
        }
        "pitch" => {
            yaw_pitch_roll_cmd_deg[1] = parse_positional_number(&tokens, 1, "pitch")?;
        }
        "roll" => {
            yaw_pitch_roll_cmd_deg[2] = parse_positional_number(&tokens, 1, "roll")?;
        }
        "drive" | "command" => {
            yaw_pitch_roll_cmd_deg = [
                fields.number(&["yaw", "yaw_rate"])?.unwrap_or(0.0),
                fields.number(&["pitch", "pitch_rate"])?.unwrap_or(0.0),
                fields.number(&["roll", "roll_rate"])?.unwrap_or(0.0),
            ];
            body_cmd = [
                fields
                    .number(&["ax", "accel", "forward_accel"])?
                    .unwrap_or(0.0),
                fields.number(&["ay", "lateral_accel"])?.unwrap_or(0.0),
                fields.number(&["az", "vertical_accel"])?.unwrap_or(0.0),
            ];
        }
        other => bail!("unknown motion DSL command '{other}'"),
    }

    Ok(MotionCommand {
        command_type: 1,
        yaw_pitch_roll_cmd_deg,
        body_cmd,
        duration_s,
        gps_visible,
    })
}

fn parse_duration(tokens: &[&str], fields: &KeyValues<'_>) -> Result<f64> {
    if let Some(duration) = fields.number(&["duration", "dur", "for"])? {
        return Ok(duration);
    }
    for window in tokens.windows(2) {
        if window[0] == "for" {
            return parse_number(window[1]);
        }
    }
    for token in tokens.iter().skip(1) {
        if token.contains('s') && !token.contains('=') {
            return parse_number(token);
        }
    }
    bail!("motion command requires a duration, e.g. 'for 8s'")
}

fn parse_gps_visibility(tokens: &[&str], fields: &KeyValues<'_>) -> Result<bool> {
    if let Some(value) = fields.raw(&["gps", "gnss"]) {
        return parse_bool(value);
    }
    if tokens.windows(2).any(|window| {
        matches!(window[0], "gps" | "gnss") && matches!(window[1], "off" | "false" | "0")
    }) {
        return Ok(false);
    }
    Ok(!tokens
        .iter()
        .any(|token| matches!(*token, "no_gps" | "no_gnss")))
}

fn parse_positional_number(tokens: &[&str], start: usize, command: &str) -> Result<f64> {
    tokens
        .iter()
        .skip(start)
        .find(|token| !matches!(**token, "for" | "gps" | "gnss") && !token.contains('='))
        .ok_or_else(|| anyhow::anyhow!("{command} requires a numeric value"))
        .and_then(|token| parse_number(token))
}

fn parse_bool(value: &str) -> Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "on" | "yes" | "1" => Ok(true),
        "false" | "off" | "no" | "0" => Ok(false),
        other => bail!("invalid boolean value '{other}'"),
    }
}

fn parse_number(token: &str) -> Result<f64> {
    let token = token.trim().trim_matches(',');
    let end = token
        .char_indices()
        .take_while(|(_, ch)| ch.is_ascii_digit() || matches!(*ch, '+' | '-' | '.' | 'e' | 'E'))
        .map(|(idx, ch)| idx + ch.len_utf8())
        .last()
        .unwrap_or(0);
    if end == 0 {
        bail!("expected numeric value, got '{token}'");
    }
    token[..end]
        .parse::<f64>()
        .map_err(|_| anyhow::anyhow!("invalid numeric value '{token}'"))
}

fn first_token(line: &str) -> Option<&str> {
    line.split_whitespace().next()
}

fn split_tokens(line: &str) -> Vec<&str> {
    line.split(|ch: char| ch.is_whitespace() || ch == ',')
        .map(|token| token.trim())
        .filter(|token| !token.is_empty())
        .collect()
}

struct KeyValues<'a> {
    values: Vec<(&'a str, &'a str)>,
}

impl<'a> KeyValues<'a> {
    fn from_line(line: &'a str) -> Self {
        let tokens = split_tokens(line);
        Self::from_tokens(&tokens)
    }

    fn from_tokens(tokens: &[&'a str]) -> Self {
        let values = tokens
            .iter()
            .filter_map(|token| {
                token
                    .split_once('=')
                    .map(|(key, value)| (key.trim(), value.trim()))
            })
            .collect();
        Self { values }
    }

    fn raw(&self, names: &[&str]) -> Option<&'a str> {
        self.values
            .iter()
            .find(|(key, _)| names.iter().any(|name| key.eq_ignore_ascii_case(name)))
            .map(|(_, value)| *value)
    }

    fn number(&self, names: &[&str]) -> Result<Option<f64>> {
        self.raw(names).map(parse_number).transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_high_level_city_block_style_scenario() -> Result<()> {
        let profile = parse_motion_dsl(
            r#"
            initial lat=32 lon=120 alt=0 speed=0 yaw=0 pitch=0 roll=0
            wait 60s
            repeat 2 {
                accelerate 1.0m/s^2 for 8s
                hold 12s
                turn left 10dps for 9s
                hold 12s
                brake 1.0mps2 for 8s
                coast 11s
                drive yaw=-10dps ax=0 for=9s gps=off
            }
            "#,
        )?;
        assert_eq!(profile.initial.lat_deg, 32.0);
        assert_eq!(profile.commands.len(), 15);
        assert_eq!(profile.commands[1].body_cmd[0], 1.0);
        assert_eq!(profile.commands[3].yaw_pitch_roll_cmd_deg[0], 10.0);
        assert_eq!(profile.commands[5].body_cmd[0], -1.0);
        assert_eq!(profile.commands[7].yaw_pitch_roll_cmd_deg[0], -10.0);
        assert!(!profile.commands[7].gps_visible);
        Ok(())
    }
}
