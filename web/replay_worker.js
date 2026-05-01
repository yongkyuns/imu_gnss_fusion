let wasmReady = null;
let visualizer = null;

async function ensureWasm() {
  if (!wasmReady) {
    const version = new URL(self.location.href).searchParams.get("v") || String(Date.now());
    wasmReady = import(`./pkg/visualizer.js?v=${encodeURIComponent(version)}`).then(async (module) => {
      await module.default({
        module_or_path: `./pkg/visualizer_bg.wasm?v=${encodeURIComponent(version)}`,
      });
      visualizer = module;
    });
  }
  await wasmReady;
}

self.onmessage = async (event) => {
  const request = normalizeReplayJob(event.data || {});
  try {
    await ensureWasm();
    self.postMessage(runReplayJob(request));
  } catch (error) {
    self.postMessage({
      ok: false,
      jobId: request.jobId ?? 0,
      label: request.label || "CSV replay",
      error: error && error.message ? error.message : String(error),
    });
  }
};

function normalizeReplayJob(request) {
  if (request.source) {
    return request;
  }
  return {
    ...request,
    source: {
      kind: "csv",
      label: request.label || "CSV replay",
      imuName: request.imuName || "imu.csv",
      gnssName: request.gnssName || "gnss.csv",
      imuCsv: request.imuCsv || "",
      gnssCsv: request.gnssCsv || "",
      referenceAttitudeCsv: request.referenceAttitudeCsv ?? null,
      referenceMountCsv: request.referenceMountCsv ?? null,
    },
  };
}

function runReplayJob(request) {
  if (visualizer.build_replay_job_json_with_progress) {
    const progress = (message) => {
      self.postMessage(message);
    };
    return JSON.parse(
      visualizer.build_replay_job_json_with_progress(JSON.stringify(request), progress),
    );
  }

  if (visualizer.build_generic_replay_job_json_with_progress) {
    const progress = (message) => {
      self.postMessage(message);
    };
    return JSON.parse(
      visualizer.build_generic_replay_job_json_with_progress(
        JSON.stringify(request),
        progress,
      ),
    );
  }

  if (visualizer.build_replay_plot_data_json) {
    const json = visualizer.build_replay_plot_data_json(JSON.stringify(request));
    return replayResultFromRequest(request, json);
  }

  if (visualizer.build_replay_plot_data_json_with_progress) {
    const json = visualizer.build_replay_plot_data_json_with_progress(
      JSON.stringify(request),
      () => {},
    );
    return replayResultFromRequest(request, json);
  }

  if (visualizer.build_generic_replay_job_json) {
    return JSON.parse(visualizer.build_generic_replay_job_json(JSON.stringify(request)));
  }

  const json = visualizer.build_generic_replay_plot_data_json(
    request.imuCsv || "",
    request.gnssCsv || "",
    request.referenceAttitudeCsv ?? null,
    request.referenceMountCsv ?? null,
  );
  return {
    ok: true,
    jobId: request.jobId ?? 0,
    label: request.label || "CSV replay",
    source: "csv",
    imuName: request.imuName || "imu.csv",
    gnssName: request.gnssName || "gnss.csv",
    json,
  };
}

function replayResultFromRequest(request, json) {
  const source = request.source || {};
  const kind = source.kind || "csv";
  const isSynthetic = kind === "synthetic";
  return {
    ok: true,
    jobId: request.jobId ?? 0,
    label: source.label || request.label || (isSynthetic ? "Synthetic replay" : "CSV replay"),
    source: isSynthetic ? "synthetic" : "csv",
    imuName: source.imuName || request.imuName || (isSynthetic ? "synthetic IMU" : "imu.csv"),
    gnssName: source.gnssName || request.gnssName || (isSynthetic ? "synthetic GNSS" : "gnss.csv"),
    json,
  };
}
