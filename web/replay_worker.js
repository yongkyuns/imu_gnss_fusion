import init, * as visualizer from "./pkg/visualizer.js";

let wasmReady = null;

async function ensureWasm() {
  if (!wasmReady) {
    wasmReady = init();
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
