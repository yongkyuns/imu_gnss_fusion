#!/usr/bin/env node

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MIME_TYPES = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".mjs", "text/javascript; charset=utf-8"],
  [".wasm", "application/wasm"],
  [".css", "text/css; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
]);

function usage() {
  console.log(`Usage: node scripts/validate_pages_static.mjs [options]

Validate the static GitHub Pages visualizer artifact before upload/deploy.

Options:
  --site-dir <dir>    Static site directory (default: web)
  --require-wasm      Require web/pkg/visualizer.js and visualizer_bg.wasm
  --help              Show this help
`);
}

function parseArgs(argv) {
  const args = {
    siteDir: path.join(ROOT, "web"),
    requireWasm: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      i += 1;
      if (i >= argv.length) throw new Error(`missing value for ${arg}`);
      return argv[i];
    };
    switch (arg) {
      case "--site-dir":
        args.siteDir = path.resolve(ROOT, next());
        break;
      case "--require-wasm":
        args.requireWasm = true;
        break;
      case "--help":
      case "-h":
        usage();
        process.exit(0);
      default:
        throw new Error(`unknown argument: ${arg}`);
    }
  }
  return args;
}

async function validateFiles(siteDir, requireWasm) {
  const indexPath = path.join(siteDir, "index.html");
  const index = await readFile(indexPath, "utf8");
  if (!index.includes('id="visualizer_canvas"')) {
    throw new Error("web/index.html is missing #visualizer_canvas");
  }
  if (!index.includes("./pkg/visualizer.js")) {
    throw new Error("web/index.html does not load ./pkg/visualizer.js relatively");
  }
  if (/file:\/\/|\/Users\/|C:\\\\/.test(index)) {
    throw new Error("web/index.html contains a local filesystem reference");
  }

  if (requireWasm) {
    const jsPath = path.join(siteDir, "pkg", "visualizer.js");
    const wasmPath = path.join(siteDir, "pkg", "visualizer_bg.wasm");
    for (const filePath of [jsPath, wasmPath]) {
      if (!existsSync(filePath)) throw new Error(`missing ${filePath}`);
      const info = await stat(filePath);
      if (info.size <= 0) throw new Error(`${filePath} is empty`);
    }
    const wasm = await readFile(wasmPath);
    if (wasm[0] !== 0x00 || wasm[1] !== 0x61 || wasm[2] !== 0x73 || wasm[3] !== 0x6d) {
      throw new Error("visualizer_bg.wasm does not have a wasm magic header");
    }
  }

  const datasetManifestPath = path.join(siteDir, "datasets", "manifest.json");
  if (existsSync(datasetManifestPath)) {
    validateDatasetManifest(JSON.parse(await readFile(datasetManifestPath, "utf8")), datasetManifestPath);
  }
}

function validateDatasetManifest(manifest, manifestPath) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    throw new Error(`${manifestPath} must contain a JSON object`);
  }
  if (!Array.isArray(manifest.datasets)) {
    throw new Error(`${manifestPath} must contain a datasets array`);
  }
  for (const [index, dataset] of manifest.datasets.entries()) {
    if (!dataset || typeof dataset !== "object" || Array.isArray(dataset)) {
      throw new Error(`${manifestPath}: datasets[${index}] must be an object`);
    }
    for (const key of [
      "base_url",
      "baseUrl",
      "imu",
      "gnss",
      "imu_gz",
      "gnss_gz",
      "imu_csv",
      "gnss_csv",
      "reference_attitude",
      "reference_attitude_gz",
      "reference_attitude_csv",
      "reference_attitude_csv_gz",
      "reference_mount",
      "reference_mount_gz",
      "reference_mount_csv",
      "reference_mount_csv_gz",
    ]) {
      if (dataset[key] !== undefined && !isSafeDatasetUrl(dataset[key])) {
        throw new Error(`${manifestPath}: datasets[${index}].${key} must be relative or HTTPS`);
      }
    }
  }
}

function isSafeDatasetUrl(value) {
  if (typeof value !== "string" || value.length === 0) return false;
  if (/^https:\/\//.test(value)) return true;
  if (/^[a-z]+:\/\//i.test(value) || value.startsWith("/") || value.includes("..")) return false;
  return !/file:\/\/|\/Users\/|C:\\\\/.test(value);
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
      res.writeHead(200, {
        "Content-Type": MIME_TYPES.get(path.extname(filePath)) || "application/octet-stream",
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
    port: server.address().port,
    close: () => new Promise((resolve) => server.close(resolve)),
  };
}

async function validateHttp(siteDir, requireWasm) {
  const server = await startStaticServer(siteDir);
  try {
    await expectOk(`http://127.0.0.1:${server.port}/index.html`, "text/html");
    if (requireWasm) {
      await expectOk(`http://127.0.0.1:${server.port}/pkg/visualizer.js`, "text/javascript");
      await expectOk(`http://127.0.0.1:${server.port}/pkg/visualizer_bg.wasm`, "application/wasm");
    }
    await validateDatasetHttp(`http://127.0.0.1:${server.port}`);
  } finally {
    await server.close();
  }
}

async function validateDatasetHttp(origin) {
  const manifestUrl = `${origin}/datasets/manifest.json`;
  const response = await fetch(manifestUrl);
  if (response.status === 404) {
    return;
  }
  if (!response.ok) throw new Error(`${manifestUrl} returned HTTP ${response.status}`);
  const manifest = await response.json();
  validateDatasetManifest(manifest, manifestUrl);
  for (const [index, dataset] of manifest.datasets.entries()) {
    for (const key of [
      "imu_gz",
      "gnss_gz",
      "reference_attitude_gz",
      "reference_mount_gz",
      "imu",
      "gnss",
      "reference_attitude",
      "reference_mount",
      "imu_csv",
      "gnss_csv",
      "reference_attitude_csv",
      "reference_mount_csv",
    ]) {
      if (!dataset[key]) continue;
      const url = new URL(
        dataset[key],
        `${origin}/datasets/${dataset.base_url || dataset.baseUrl || ""}/`,
      ).toString();
      const fileResponse = await fetch(url);
      if (!fileResponse.ok) {
        throw new Error(`${manifestUrl}: datasets[${index}].${key} ${url} returned HTTP ${fileResponse.status}`);
      }
      const body = Buffer.from(await fileResponse.arrayBuffer());
      if (body.length === 0) {
        throw new Error(`${url} is empty`);
      }
      if (url.endsWith(".gz") && (body[0] !== 0x1f || body[1] !== 0x8b)) {
        throw new Error(`${url} is not a gzip file`);
      }
    }
  }
}

async function expectOk(url, contentTypePrefix) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}`);
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.startsWith(contentTypePrefix)) {
    throw new Error(`${url} content-type ${contentType} did not start with ${contentTypePrefix}`);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  await validateFiles(args.siteDir, args.requireWasm);
  await validateHttp(args.siteDir, args.requireWasm);
  console.log(`pages static artifact ok: ${args.siteDir}`);
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
