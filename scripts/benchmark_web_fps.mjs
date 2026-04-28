#!/usr/bin/env node

import { createServer } from "node:http";
import { readFile, rm } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const DEFAULT_BROWSER_PATHS = [
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
];

const MIME_TYPES = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".mjs", "text/javascript; charset=utf-8"],
  [".wasm", "application/wasm"],
  [".css", "text/css; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
]);

function usage() {
  console.log(`Usage: node scripts/benchmark_web_fps.mjs [options]

Launch the built wasm visualizer in Chrome and collect requestAnimationFrame timing.

Options:
  --serve-dir <dir>       Static directory to serve (default: web)
  --url-path <path>       Path to open from the served directory (default: /index.html)
  --browser <path>        Chrome/Chromium executable (or set BROWSER)
  --warmup-ms <n>         Warmup after visualizer startup before sampling (default: 2000)
  --duration-ms <n>       Sampling duration (default: 10000)
  --width <n>             Viewport width in CSS pixels (default: 1280)
  --height <n>            Viewport height in CSS pixels (default: 800)
  --device-scale-factor <n>
                          Browser device scale factor / DPR (default: 1)
  --scenario <name>       Built-in visualizer scenario for ?bench=1 (default: city_blocks)
  --dataset <id>          Hosted dataset id to auto-load with ?dataset=<id>
  --activity <mode>       none or mouse; mouse moves over the canvas during sampling (default: mouse)
  --min-fps <n>           Exit non-zero if mean FPS is below this threshold
  --json                  Print the full JSON result instead of a short text summary
  --help                  Show this help
`);
}

function parseArgs(argv) {
  const args = {
    serveDir: path.join(ROOT, "web"),
    urlPath: "/index.html",
    browser: process.env.BROWSER || "",
    warmupMs: 2000,
    durationMs: 10000,
    width: 1280,
    height: 800,
    deviceScaleFactor: 1,
    scenario: "city_blocks",
    dataset: "",
    activity: "mouse",
    minFps: null,
    json: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      i += 1;
      if (i >= argv.length) {
        throw new Error(`missing value for ${arg}`);
      }
      return argv[i];
    };

    switch (arg) {
      case "--serve-dir":
        args.serveDir = path.resolve(ROOT, next());
        break;
      case "--url-path":
        args.urlPath = next();
        break;
      case "--browser":
        args.browser = next();
        break;
      case "--warmup-ms":
        args.warmupMs = parsePositiveInt(arg, next());
        break;
      case "--duration-ms":
        args.durationMs = parsePositiveInt(arg, next());
        break;
      case "--width":
        args.width = parsePositiveInt(arg, next());
        break;
      case "--height":
        args.height = parsePositiveInt(arg, next());
        break;
      case "--device-scale-factor":
        args.deviceScaleFactor = parsePositiveFloat(arg, next());
        break;
      case "--scenario":
        args.scenario = next();
        break;
      case "--dataset":
        args.dataset = next();
        break;
      case "--activity":
        args.activity = next();
        if (!["none", "mouse"].includes(args.activity)) {
          throw new Error("--activity must be either none or mouse");
        }
        break;
      case "--min-fps":
        args.minFps = parsePositiveFloat(arg, next());
        break;
      case "--json":
        args.json = true;
        break;
      case "--help":
      case "-h":
        usage();
        process.exit(0);
        break;
      default:
        throw new Error(`unknown argument: ${arg}`);
    }
  }

  if (!args.urlPath.startsWith("/")) {
    args.urlPath = `/${args.urlPath}`;
  }

  return args;
}

function parsePositiveInt(name, value) {
  const n = Number.parseInt(value, 10);
  if (!Number.isFinite(n) || n <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return n;
}

function parsePositiveFloat(name, value) {
  const n = Number.parseFloat(value);
  if (!Number.isFinite(n) || n <= 0) {
    throw new Error(`${name} must be a positive number`);
  }
  return n;
}

function findBrowser(browserArg) {
  if (browserArg && existsSync(browserArg)) {
    return browserArg;
  }
  for (const candidate of DEFAULT_BROWSER_PATHS) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  throw new Error(
    "Chrome/Chromium was not found. Install Google Chrome/Chromium or pass --browser <path>.",
  );
}

async function startStaticServer(rootDir) {
  const server = createServer(async (req, res) => {
    try {
      const requestPath = new URL(req.url || "/", "http://localhost").pathname;
      const normalized = path.normalize(decodeURIComponent(requestPath)).replace(/^(\.\.[/\\])+/, "");
      const filePath = path.join(rootDir, normalized === "/" ? "index.html" : normalized);
      const relative = path.relative(rootDir, filePath);
      if (relative.startsWith("..") || path.isAbsolute(relative)) {
        res.writeHead(403).end("forbidden");
        return;
      }
      const body = await readFile(filePath);
      const mime = MIME_TYPES.get(path.extname(filePath)) || "application/octet-stream";
      res.writeHead(200, {
        "Content-Type": mime,
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
      });
      res.end(body);
    } catch (error) {
      res.writeHead(error?.code === "ENOENT" ? 404 : 500).end(String(error));
    }
  });

  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });

  return {
    server,
    port: server.address().port,
    close: () => new Promise((resolve) => server.close(resolve)),
  };
}

async function findFreePort() {
  const server = createServer();
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const { port } = server.address();
  await new Promise((resolve) => server.close(resolve));
  return port;
}

async function httpJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`${options.method || "GET"} ${url} failed with HTTP ${response.status}`);
  }
  return response.json();
}

async function waitForHttpJson(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      return await httpJson(url);
    } catch (error) {
      lastError = error;
      await delay(100);
    }
  }
  throw new Error(`timed out waiting for ${url}: ${lastError}`);
}

class CdpClient {
  constructor(wsUrl) {
    this.nextId = 1;
    this.pending = new Map();
    this.eventWaiters = [];
    this.consoleMessages = [];
    this.exceptions = [];
    this.ws = new WebSocket(wsUrl);
  }

  async open() {
    await new Promise((resolve, reject) => {
      this.ws.addEventListener("open", resolve, { once: true });
      this.ws.addEventListener("error", reject, { once: true });
    });
    this.ws.addEventListener("message", (event) => this.#handleMessage(event));
    this.ws.addEventListener("close", () => {
      for (const { reject, timeout } of this.pending.values()) {
        clearTimeout(timeout);
        reject(new Error("CDP websocket closed"));
      }
      this.pending.clear();
    });
  }

  async close() {
    this.ws.close();
  }

  send(method, params = {}, timeoutMs = 15000) {
    const id = this.nextId;
    this.nextId += 1;
    const message = JSON.stringify({ id, method, params });
    const promise = new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`CDP ${method} timed out`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timeout, method });
    });
    this.ws.send(message);
    return promise;
  }

  waitForEvent(method, timeoutMs = 15000) {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.eventWaiters = this.eventWaiters.filter((waiter) => waiter !== waiterRecord);
        reject(new Error(`timed out waiting for CDP event ${method}`));
      }, timeoutMs);
      const waiterRecord = { method, resolve, reject, timeout };
      this.eventWaiters.push(waiterRecord);
    });
  }

  #handleMessage(event) {
    const message = JSON.parse(event.data);
    if (message.id) {
      const pending = this.pending.get(message.id);
      if (!pending) {
        return;
      }
      clearTimeout(pending.timeout);
      this.pending.delete(message.id);
      if (message.error) {
        pending.reject(new Error(`CDP ${pending.method} failed: ${message.error.message}`));
      } else {
        pending.resolve(message.result || {});
      }
      return;
    }

    if (message.method === "Runtime.consoleAPICalled") {
      this.consoleMessages.push({
        type: message.params.type,
        text: (message.params.args || []).map((arg) => arg.value ?? arg.description ?? "").join(" "),
      });
    } else if (message.method === "Runtime.exceptionThrown") {
      this.exceptions.push(message.params.exceptionDetails?.text || "Runtime exception");
    }

    for (const waiter of [...this.eventWaiters]) {
      if (waiter.method === message.method) {
        clearTimeout(waiter.timeout);
        this.eventWaiters = this.eventWaiters.filter((candidate) => candidate !== waiter);
        waiter.resolve(message.params || {});
      }
    }
  }
}

async function evaluate(cdp, expression, timeoutMs = 15000) {
  const result = await cdp.send(
    "Runtime.evaluate",
    {
      expression,
      awaitPromise: true,
      returnByValue: true,
      userGesture: true,
    },
    timeoutMs,
  );
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.text || "Runtime.evaluate failed");
  }
  return result.result?.value;
}

async function waitForExpression(cdp, expression, timeoutMs, label) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = await evaluate(cdp, expression);
    if (value) {
      return value;
    }
    await delay(100);
  }
  throw new Error(`timed out waiting for ${label}`);
}

async function launchChrome(browserPath, debugPort, userDataDir, width, height, deviceScaleFactor) {
  const child = spawn(
    browserPath,
    [
      "--headless=new",
      "--remote-debugging-address=127.0.0.1",
      `--remote-debugging-port=${debugPort}`,
      `--user-data-dir=${userDataDir}`,
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-background-timer-throttling",
      "--disable-renderer-backgrounding",
      "--disable-backgrounding-occluded-windows",
      "--disable-features=CalculateNativeWinOcclusion",
      `--force-device-scale-factor=${deviceScaleFactor}`,
      `--window-size=${width},${height}`,
      "about:blank",
    ],
    { stdio: ["ignore", "ignore", "pipe"] },
  );

  let stderr = "";
  child.stderr.on("data", (chunk) => {
    stderr += chunk.toString();
  });

  child.on("exit", (code, signal) => {
    if (code !== 0 && signal !== "SIGTERM") {
      console.error(`Chrome exited with code=${code} signal=${signal}\n${stderr.trim()}`);
    }
  });

  return child;
}

function installProbeSource() {
  return `
(() => {
  if (window.__visualizerBenchmark) return;
  const originalRequestAnimationFrame = window.requestAnimationFrame.bind(window);
  const stats = {
    installedAtMs: performance.now(),
    resetAtMs: null,
    frameTimestampsMs: [],
    callbackDurationsMs: [],
    callbackCount: 0,
    callbackErrors: [],
  };

  window.__visualizerBenchmark = {
    reset() {
      stats.resetAtMs = performance.now();
      stats.frameTimestampsMs = [];
      stats.callbackDurationsMs = [];
      stats.callbackCount = 0;
      stats.callbackErrors = [];
    },
    snapshot() {
      return {
        installedAtMs: stats.installedAtMs,
        resetAtMs: stats.resetAtMs,
        sampledAtMs: performance.now(),
        frameTimestampsMs: stats.frameTimestampsMs.slice(),
        callbackDurationsMs: stats.callbackDurationsMs.slice(),
        callbackCount: stats.callbackCount,
        callbackErrors: stats.callbackErrors.slice(),
        appPerf: window.__imuGnssFusionPerf || null,
      };
    },
  };

  window.requestAnimationFrame = (callback) => originalRequestAnimationFrame((timestamp) => {
    stats.frameTimestampsMs.push(timestamp);
    stats.callbackCount += 1;
    const callbackStart = performance.now();
    try {
      return callback(timestamp);
    } catch (error) {
      stats.callbackErrors.push(String(error && error.stack ? error.stack : error));
      throw error;
    } finally {
      stats.callbackDurationsMs.push(performance.now() - callbackStart);
    }
  });
})();
`;
}

async function driveMouse(cdp, width, height, durationMs) {
  const start = Date.now();
  const centerY = Math.round(height * 0.55);
  const radiusX = Math.max(20, Math.round(width * 0.32));
  const centerX = Math.round(width * 0.5);
  let i = 0;

  while (Date.now() - start < durationMs) {
    const phase = i / 18;
    const x = Math.round(centerX + Math.cos(phase) * radiusX);
    const y = Math.round(centerY + Math.sin(phase * 0.7) * Math.max(20, height * 0.18));
    await cdp.send("Input.dispatchMouseEvent", {
      type: "mouseMoved",
      x: clamp(x, 1, width - 2),
      y: clamp(y, 1, height - 2),
      button: "none",
      buttons: 0,
    });
    i += 1;
    await delay(16);
  }
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function stopProcess(child) {
  if (child.exitCode !== null || child.signalCode !== null) {
    return;
  }
  child.kill("SIGTERM");
  const exited = await Promise.race([
    new Promise((resolve) => child.once("exit", () => resolve(true))),
    delay(3000).then(() => false),
  ]);
  if (!exited) {
    child.kill("SIGKILL");
    await Promise.race([
      new Promise((resolve) => child.once("exit", resolve)),
      delay(1000),
    ]);
  }
}

function summarize(raw, options, pageInfo, consoleMessages, exceptions) {
  const uniqueFrames = [];
  for (const timestamp of raw.frameTimestampsMs || []) {
    const last = uniqueFrames[uniqueFrames.length - 1];
    if (last === undefined || Math.abs(timestamp - last) > 0.01) {
      uniqueFrames.push(timestamp);
    }
  }

  const intervals = [];
  for (let i = 1; i < uniqueFrames.length; i += 1) {
    intervals.push(uniqueFrames[i] - uniqueFrames[i - 1]);
  }

  const observedDurationMs =
    uniqueFrames.length >= 2 ? uniqueFrames[uniqueFrames.length - 1] - uniqueFrames[0] : 0;
  const meanFrameMs = mean(intervals);
  const fps = meanFrameMs > 0 ? 1000 / meanFrameMs : 0;
  const callbackDurations = raw.callbackDurationsMs || [];

  return {
    url: options.url,
    viewport: {
      width: options.width,
      height: options.height,
      deviceScaleFactor: options.deviceScaleFactor,
    },
    warmupMs: options.warmupMs,
    requestedDurationMs: options.durationMs,
    activity: options.activity,
    page: pageInfo,
    frames: {
      callbackCount: raw.callbackCount || 0,
      uniqueFrameCount: uniqueFrames.length,
      observedDurationMs: round(observedDurationMs),
      fps: round(fps),
      meanFrameMs: round(meanFrameMs),
      minFrameMs: round(min(intervals)),
      p50FrameMs: round(percentile(intervals, 50)),
      p95FrameMs: round(percentile(intervals, 95)),
      p99FrameMs: round(percentile(intervals, 99)),
      maxFrameMs: round(max(intervals)),
      framesOver33ms: intervals.filter((value) => value > 33.333).length,
      framesOver50ms: intervals.filter((value) => value > 50).length,
    },
    callbacks: {
      meanDurationMs: round(mean(callbackDurations)),
      p95DurationMs: round(percentile(callbackDurations, 95)),
      maxDurationMs: round(max(callbackDurations)),
      errors: raw.callbackErrors || [],
    },
    egui: raw.appPerf
      ? {
          frameCount: raw.appPerf.frameCount,
          elapsedSec: round(raw.appPerf.elapsedSec),
          avgFps: round(raw.appPerf.avgFps),
          emaFps: round(raw.appPerf.emaFps),
          maxPointsPerTrace: raw.appPerf.maxPointsPerTrace,
        }
      : null,
    diagnostics: {
      consoleMessages,
      exceptions,
    },
  };
}

function mean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function min(values) {
  return values.length ? Math.min(...values) : 0;
}

function max(values) {
  return values.length ? Math.max(...values) : 0;
}

function percentile(values, p) {
  if (!values.length) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const index = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) {
    return sorted[lower];
  }
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

function round(value) {
  return Number.isFinite(value) ? Math.round(value * 100) / 100 : 0;
}

function printSummary(result) {
  console.log("Web visualizer FPS benchmark");
  console.log(`  URL: ${result.url}`);
  console.log(
    `  Viewport: ${result.viewport.width}x${result.viewport.height} @ ${result.viewport.deviceScaleFactor}x DPR`,
  );
  console.log(`  Activity: ${result.activity}`);
  console.log(
    `  Frames: ${result.frames.uniqueFrameCount} unique rAF frames over ${result.frames.observedDurationMs} ms`,
  );
  console.log(
    `  FPS: ${result.frames.fps} mean (${result.frames.meanFrameMs} ms/frame), p95 ${result.frames.p95FrameMs} ms, p99 ${result.frames.p99FrameMs} ms`,
  );
  console.log(
    `  Long frames: ${result.frames.framesOver33ms} >33.3 ms, ${result.frames.framesOver50ms} >50 ms`,
  );
  console.log(
    `  rAF callback duration: mean ${result.callbacks.meanDurationMs} ms, p95 ${result.callbacks.p95DurationMs} ms, max ${result.callbacks.maxDurationMs} ms`,
  );
  if (result.egui) {
    console.log(
      `  egui frames: ${result.egui.frameCount} over ${result.egui.elapsedSec}s, avg ${result.egui.avgFps} FPS, EMA ${result.egui.emaFps} FPS`,
    );
  }
  if (result.diagnostics.exceptions.length || result.callbacks.errors.length) {
    console.log("  Diagnostics: runtime exceptions or rAF callback errors were captured; rerun with --json.");
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const browserPath = findBrowser(args.browser);

  if (!existsSync(path.join(args.serveDir, "pkg", "visualizer.js"))) {
    throw new Error(
      `missing ${path.join(args.serveDir, "pkg", "visualizer.js")}; build the wasm bundle first`,
    );
  }
  if (!existsSync(path.join(args.serveDir, "pkg", "visualizer_bg.wasm"))) {
    throw new Error(
      `missing ${path.join(args.serveDir, "pkg", "visualizer_bg.wasm")}; build the wasm bundle first`,
    );
  }

  const server = await startStaticServer(args.serveDir);
  const debugPort = await findFreePort();
  const userDataDir = path.join(tmpdir(), `imu-gnss-web-bench-${process.pid}`);
  const browser = await launchChrome(
    browserPath,
    debugPort,
    userDataDir,
    args.width,
    args.height,
    args.deviceScaleFactor,
  );
  let cdp;

  try {
    await waitForHttpJson(`http://127.0.0.1:${debugPort}/json/version`, 10000);
    const targets = await httpJson(`http://127.0.0.1:${debugPort}/json/list`);
    const pageTarget = targets.find((target) => target.type === "page" && target.webSocketDebuggerUrl);
    if (!pageTarget) {
      throw new Error("Chrome did not expose a page target over DevTools");
    }

    cdp = new CdpClient(pageTarget.webSocketDebuggerUrl);
    await cdp.open();
    await cdp.send("Runtime.enable");
    await cdp.send("Page.enable");
    await cdp.send("Input.setIgnoreInputEvents", { ignore: false });
    await cdp.send("Page.addScriptToEvaluateOnNewDocument", { source: installProbeSource() });

    const url = `http://127.0.0.1:${server.port}${withBenchQuery(args.urlPath, args.scenario, args.dataset)}`;
    args.url = url;
    const loadEvent = cdp.waitForEvent("Page.loadEventFired", 20000);
    await cdp.send("Page.navigate", { url });
    await loadEvent;

    const ready = await waitForExpression(
      cdp,
      `(() => {
        const status = document.getElementById("status");
        if (!status) return { ok: true, statusText: null };
        const text = status.textContent || "";
        if (text.startsWith("Failed to start visualizer")) return { ok: false, statusText: text };
        return false;
      })()`,
      30000,
      "visualizer startup",
    );

    if (!ready.ok) {
      throw new Error(ready.statusText || "visualizer failed to start");
    }

    if (args.dataset) {
      const datasetReady = await waitForExpression(
        cdp,
        `(() => {
          const perf = window.__imuGnssFusionPerf;
          const status = perf && perf.status ? String(perf.status) : "";
          if (status.startsWith("Dataset loaded:")) return { ok: true, statusText: status };
          if (status.startsWith("Dataset load failed:") || status.startsWith("Dataset fetched but")) {
            return { ok: false, statusText: status };
          }
          return false;
        })()`,
        30000,
        "dataset auto-load",
      );
      if (!datasetReady.ok) {
        throw new Error(datasetReady.statusText || "dataset auto-load failed");
      }
    }

    await evaluate(
      cdp,
      `new Promise((resolve) => setTimeout(resolve, ${JSON.stringify(args.warmupMs)}))`,
      args.warmupMs + 5000,
    );
    await evaluate(cdp, "window.__visualizerBenchmark.reset()");

    const activity =
      args.activity === "mouse" ? driveMouse(cdp, args.width, args.height, args.durationMs) : delay(args.durationMs);
    const sampleTimer = evaluate(
      cdp,
      `new Promise((resolve) => setTimeout(resolve, ${JSON.stringify(args.durationMs)}))`,
      args.durationMs + 5000,
    );
    await Promise.all([activity, sampleTimer]);

    const raw = await evaluate(cdp, "window.__visualizerBenchmark.snapshot()", 5000);
    const pageInfo = await evaluate(
      cdp,
      `(() => {
        const nav = performance.getEntriesByType("navigation")[0];
        const canvas = document.getElementById("visualizer_canvas");
        return {
          title: document.title,
          navigation: nav ? {
            domContentLoadedEventEndMs: Math.round(nav.domContentLoadedEventEnd * 100) / 100,
            loadEventEndMs: Math.round(nav.loadEventEnd * 100) / 100,
          } : null,
          canvas: canvas ? {
            clientWidth: canvas.clientWidth,
            clientHeight: canvas.clientHeight,
            width: canvas.width,
            height: canvas.height,
          } : null,
          mapDiagnostics: window.__imuGnssFusionMapDiagnostics || null,
          hasInitialMapboxToken: Boolean(
            window.__imuGnssFusionInitialMapboxToken && window.__imuGnssFusionInitialMapboxToken(),
          ),
        };
      })()`,
    );

    const result = summarize(raw, args, pageInfo, cdp.consoleMessages, cdp.exceptions);
    if (args.json) {
      console.log(JSON.stringify(result, null, 2));
    } else {
      printSummary(result);
    }

    const measuredFps = result.egui ? Math.min(result.frames.fps, result.egui.emaFps) : result.frames.fps;
    if (args.minFps !== null && measuredFps < args.minFps) {
      process.exitCode = 2;
    }
  } finally {
    if (cdp) {
      await cdp.close();
    }
    await stopProcess(browser);
    await server.close();
    await rm(userDataDir, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(`benchmark failed: ${error.message}`);
  process.exit(1);
});

function withBenchQuery(urlPath, scenario, dataset) {
  const url = new URL(urlPath, "http://localhost");
  if (!url.searchParams.has("bench")) {
    url.searchParams.set("bench", "1");
  }
  if (dataset && !url.searchParams.has("dataset")) {
    url.searchParams.set("dataset", dataset);
    url.searchParams.delete("scenario");
  } else if (!url.searchParams.has("scenario")) {
    url.searchParams.set("scenario", scenario);
  }
  return `${url.pathname}${url.search}`;
}
