#!/usr/bin/env node

import { createServer } from "node:http";
import { readdir, readFile, stat } from "node:fs/promises";
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
  [".png", "image/png"],
  [".svg", "image/svg+xml"],
  [".ico", "image/x-icon"],
  [".pdf", "application/pdf"],
  [".txt", "text/plain; charset=utf-8"],
  [".woff", "font/woff"],
  [".woff2", "font/woff2"],
]);

function usage() {
  console.log(`Usage: node scripts/validate_pages_static.mjs [options]

Validate the static GitHub Pages visualizer artifact before upload/deploy.

Options:
  --site-dir <dir>    Static site directory (default: web)
  --docs-source-dir <dir>
                       Sphinx source directory checked with --require-docs
                       (default: docs)
  --require-wasm      Require web/pkg/visualizer.js and visualizer_bg.wasm
  --require-docs      Require Sphinx docs under docs/
  --help              Show this help
`);
}

function parseArgs(argv) {
  const args = {
    siteDir: path.join(ROOT, "web"),
    docsSourceDir: path.join(ROOT, "docs"),
    requireWasm: false,
    requireDocs: false,
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
      case "--docs-source-dir":
        args.docsSourceDir = path.resolve(ROOT, next());
        break;
      case "--require-wasm":
        args.requireWasm = true;
        break;
      case "--require-docs":
        args.requireDocs = true;
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

async function validateFiles(siteDir, requireWasm, requireDocs, docsSourceDir) {
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

  if (requireDocs) {
    await validateDocsSourceMath(docsSourceDir);

    const docsIndexPath = path.join(siteDir, "docs", "index.html");
    const docsIndex = await readFile(docsIndexPath, "utf8");
    if (!docsIndex.includes("IMU/GNSS Fusion")) {
      throw new Error("docs/index.html does not look like the IMU/GNSS Fusion docs site");
    }
    if (/file:\/\/|\/Users\/|C:\\\\/.test(docsIndex)) {
      throw new Error("docs/index.html contains a local filesystem reference");
    }
    if (!docsIndex.includes("math-overflow.js")) {
      throw new Error("docs/index.html does not load math-overflow.js");
    }

    const staticDir = path.join(siteDir, "docs", "_static");
    if (!existsSync(staticDir)) {
      throw new Error("docs/_static is missing");
    }
    await validateDocsStaticAssets(staticDir);
    await validateDocsHtml(path.join(siteDir, "docs"));
  }

  const datasetManifestPath = path.join(siteDir, "datasets", "manifest.json");
  if (existsSync(datasetManifestPath)) {
    validateDatasetManifest(JSON.parse(await readFile(datasetManifestPath, "utf8")), datasetManifestPath);
  }
}

async function validateDocsStaticAssets(staticDir) {
  const requiredAssets = [
    ["custom.css", "text"],
    ["math-overflow.js", "text"],
    ["logo.png", "png"],
    ["titlebar.png", "png"],
    ["diagrams/estimator-runtime-orthogonal.svg", "svg"],
    ["diagrams/overall-architecture-orthogonal.svg", "svg"],
    ["screenshots/web-visualizer-overview.png", "png"],
  ];
  for (const [relativePath, kind] of requiredAssets) {
    const filePath = path.join(staticDir, relativePath);
    if (!existsSync(filePath)) {
      throw new Error(`docs/_static/${relativePath} is missing`);
    }
    const info = await stat(filePath);
    if (info.size <= 0) {
      throw new Error(`docs/_static/${relativePath} is empty`);
    }
    const body = await readFile(filePath);
    if (kind === "png" && (body[0] !== 0x89 || body[1] !== 0x50 || body[2] !== 0x4e || body[3] !== 0x47)) {
      throw new Error(`docs/_static/${relativePath} is not a PNG file`);
    }
    if (kind === "svg" && !body.toString("utf8").includes("<svg")) {
      throw new Error(`docs/_static/${relativePath} does not look like an SVG file`);
    }
  }

  const mathOverflow = await readFile(path.join(staticDir, "math-overflow.js"), "utf8");
  if (!mathOverflow.includes("updateMathOverflow") || !mathOverflow.includes("math-no-scroll")) {
    throw new Error("docs/_static/math-overflow.js does not contain the expected overflow updater");
  }
}

async function validateDocsSourceMath(docsSourceDir) {
  if (!existsSync(docsSourceDir)) {
    return;
  }
  const markdownFiles = await listFiles(docsSourceDir, ".md");
  const violations = [];
  for (const filePath of markdownFiles) {
    const lines = (await readFile(filePath, "utf8")).split(/\r?\n/);
    let inFence = false;
    let inDisplayMath = false;
    for (const [index, line] of lines.entries()) {
      if (line.trimStart().startsWith("```")) {
        inFence = !inFence;
        continue;
      }
      if (!inFence && line.trim() === "$$") {
        inDisplayMath = !inDisplayMath;
        continue;
      }
      if (inFence || inDisplayMath) {
        continue;
      }
      if (/\\\(.+?\\\)|\\\[.+?\\\]|\\begin\{[^}]+\}|\\end\{[^}]+\}/.test(line)) {
        violations.push(`${path.relative(ROOT, filePath)}:${index + 1}`);
      }
    }
  }
  if (violations.length > 0) {
    throw new Error(
      `docs source uses raw LaTeX math delimiters outside display math/code fences; use MyST dollar math instead: ${violations.join(", ")}`,
    );
  }
}

async function validateDocsHtml(docsDir) {
  const htmlFiles = await listFiles(docsDir, ".html");
  const violations = [];
  for (const filePath of htmlFiles) {
    const html = await readFile(filePath, "utf8");
    if (/file:\/\/|\/Users\/|C:\\\\/.test(html)) {
      throw new Error(`${path.relative(ROOT, filePath)} contains a local filesystem reference`);
    }
    const visibleText = htmlToVisibleTextWithoutMath(html);
    const rawInlineLatex = /\\\([^)]*\\\)|\\\[[^\]]*\\\]|\\begin\{[^}]+\}|\\end\{[^}]+\}/.test(visibleText);
    const escapedInlineSymbol =
      /(^|[\s([{])\([A-Za-z\\]+_\{?[A-Za-z0-9_\\]+[^)]*\)/.test(visibleText) ||
      /(^|[\s([{])\(\\[A-Za-z]+[^)]*\)/.test(visibleText);
    if (rawInlineLatex || escapedInlineSymbol) {
      violations.push(path.relative(ROOT, filePath));
    }
    await validateDocsHtmlReferences(filePath, html, docsDir);
  }
  if (violations.length > 0) {
    throw new Error(
      `docs HTML appears to contain unrendered inline math outside MathJax spans: ${violations.join(", ")}`,
    );
  }
}

async function validateDocsHtmlReferences(filePath, html, docsDir) {
  const references = [];
  for (const match of html.matchAll(/\b(?:href|src)="([^"]+)"/g)) {
    references.push(match[1]);
  }

  for (const reference of references) {
    if (
      reference.length === 0 ||
      reference.startsWith("#") ||
      reference.startsWith("http://") ||
      reference.startsWith("https://") ||
      reference.startsWith("mailto:") ||
      reference.startsWith("javascript:")
    ) {
      continue;
    }
    const withoutFragment = reference.split("#", 1)[0].split("?", 1)[0];
    if (withoutFragment.length === 0) {
      continue;
    }
    if (withoutFragment.startsWith("/")) {
      throw new Error(`${path.relative(ROOT, filePath)} contains root-absolute docs reference ${reference}`);
    }
    const targetPath = path.resolve(path.dirname(filePath), decodeURIComponent(withoutFragment));
    const relative = path.relative(docsDir, targetPath);
    if (relative.startsWith("..") || path.isAbsolute(relative)) {
      continue;
    }
    if (!existsSync(targetPath)) {
      throw new Error(`${path.relative(ROOT, filePath)} references missing docs asset ${reference}`);
    }
    const info = await stat(targetPath);
    if (info.size <= 0) {
      throw new Error(`${path.relative(ROOT, filePath)} references empty docs asset ${reference}`);
    }
  }
}

function htmlToVisibleTextWithoutMath(html) {
  return html
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ")
    .replace(/<pre\b[^>]*>[\s\S]*?<\/pre>/gi, " ")
    .replace(/<code\b[^>]*>[\s\S]*?<\/code>/gi, " ")
    .replace(/<span\b[^>]*class="[^"]*\bmath\b[^"]*"[^>]*>[\s\S]*?<\/span>/gi, " ")
    .replace(/<div\b[^>]*class="[^"]*\bmath\b[^"]*"[^>]*>[\s\S]*?<\/div>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&")
    .replace(/&#39;/g, "'")
    .replace(/&quot;/g, '"')
    .replace(/\s+/g, " ");
}

async function listFiles(rootDir, extension) {
  const files = [];
  async function walk(dir) {
    const entries = await readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const entryPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        await walk(entryPath);
      } else if (entry.isFile() && entry.name.endsWith(extension)) {
        files.push(entryPath);
      }
    }
  }
  await walk(rootDir);
  return files;
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
      "imu_csv_gz",
      "gnss_csv_gz",
      "reference_attitude",
      "reference_attitude_gz",
      "reference_attitude_csv",
      "reference_attitude_csv_gz",
      "reference_mount",
      "reference_mount_gz",
      "reference_mount_csv",
      "reference_mount_csv_gz",
      "reference_position",
      "reference_position_gz",
      "reference_position_csv",
      "reference_position_csv_gz",
      "reference_motion",
      "reference_motion_gz",
      "reference_motion_csv",
      "reference_motion_csv_gz",
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

async function validateHttp(siteDir, requireWasm, requireDocs) {
  const server = await startStaticServer(siteDir);
  try {
    await expectOk(`http://127.0.0.1:${server.port}/index.html`, "text/html");
    if (requireWasm) {
      await expectOk(`http://127.0.0.1:${server.port}/pkg/visualizer.js`, "text/javascript");
      await expectOk(`http://127.0.0.1:${server.port}/pkg/visualizer_bg.wasm`, "application/wasm");
    }
    if (requireDocs) {
      await expectOk(`http://127.0.0.1:${server.port}/docs/index.html`, "text/html");
      await expectOk(`http://127.0.0.1:${server.port}/docs/_static/logo.png`, "image/png");
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
      "imu_csv_gz",
      "gnss_csv_gz",
      "reference_attitude_gz",
      "reference_mount_gz",
      "reference_position_gz",
      "reference_motion_gz",
      "reference_attitude_csv_gz",
      "reference_mount_csv_gz",
      "reference_position_csv_gz",
      "reference_motion_csv_gz",
      "imu",
      "gnss",
      "reference_attitude",
      "reference_mount",
      "reference_position",
      "reference_motion",
      "imu_csv",
      "gnss_csv",
      "reference_attitude_csv",
      "reference_mount_csv",
      "reference_position_csv",
      "reference_motion_csv",
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
  await validateFiles(args.siteDir, args.requireWasm, args.requireDocs, args.docsSourceDir);
  await validateHttp(args.siteDir, args.requireWasm, args.requireDocs);
  console.log(`pages static artifact ok: ${args.siteDir}`);
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
