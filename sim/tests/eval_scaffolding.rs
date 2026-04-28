use sim::eval::config::{EKF_COMPARE_DEFAULTS, snapshot_ekf_compare_config};
use sim::eval::trace::{
    find_trace, require_trace, require_trace_points, require_trace_schema, sample_nearest_point,
    sample_nearest_value,
};
use sim::visualizer::model::Trace;
use sim::visualizer::pipeline::EkfCompareConfig;

#[test]
fn trace_helpers_find_required_traces_and_sample_nearest_values() {
    let traces = vec![
        Trace {
            name: "a".to_string(),
            points: vec![[0.0, 10.0], [1.0, 20.0]],
        },
        Trace {
            name: "b".to_string(),
            points: vec![[0.0, -1.0]],
        },
    ];

    assert_eq!(find_trace(&traces, "a").expect("trace a").name, "a");
    assert!(find_trace(&traces, "missing").is_none());
    assert_eq!(
        require_trace("unit", &traces, "b").expect("trace b").points[0],
        [0.0, -1.0]
    );

    let trace = find_trace(&traces, "a").expect("trace a");
    assert_eq!(sample_nearest_point(trace, 0.75), Some([1.0, 20.0]));
    assert_eq!(sample_nearest_value(trace, 0.5), Some(10.0));
}

#[test]
fn trace_schema_helper_requires_named_traces() {
    let traces = vec![
        Trace {
            name: "posN".to_string(),
            points: vec![[0.0, 1.0]],
        },
        Trace {
            name: "posE".to_string(),
            points: vec![[0.0, 2.0]],
        },
    ];

    let required = require_trace_schema("unit", &traces, &["posN", "posE"]).expect("schema");
    assert_eq!(required.len(), 2);
    assert!(require_trace_schema("unit", &traces, &["posD"]).is_err());
    require_trace_points("unit", required[0]).expect("finite points");
}

#[test]
fn ekf_compare_config_defaults_match_regression_snapshot() {
    let actual = snapshot_ekf_compare_config(&EkfCompareConfig::default());
    assert_eq!(actual, EKF_COMPARE_DEFAULTS);
}
