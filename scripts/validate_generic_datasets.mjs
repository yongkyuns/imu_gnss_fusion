#!/usr/bin/env node

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { copyFile, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { gunzip } from "node:zlib";
import { promisify } from "node:util";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const gunzipAsync = promisify(gunzip);
const IMU_HEADER = "t_s,gx_radps,gy_radps,gz_radps,ax_mps2,ay_mps2,az_mps2";
const GNSS_HEADER =
  "t_s,lat_deg,lon_deg,height_m,vn_mps,ve_mps,vd_mps,pos_std_n_m,pos_std_e_m,pos_std_d_m,vel_std_n_mps,vel_std_e_mps,vel_std_d_mps,heading_rad";
const REFERENCE_RPY_HEADER = "t_s,roll_deg,pitch_deg,yaw_deg";

function usage() {
  console.log(`Usage: node scripts/validate_generic_datasets.mjs [options]

Validate hosted generic replay manifests, checksums, CSV schema, and optional profile smoke runs.

Options:
  --manifest <path-or-url>  Manifest JSON path or HTTPS URL (default: .github/datasets/generic-datasets.json)
  --cache-dir <dir>         Checksum-addressed download cache (default: .cache/generic-datasets)
  --work-dir <dir>          Assembled datasets and smoke subsets (default: target/generic-datasets)
  --schema-only             Validate only manifest shape, not downloads/checksums/CSV files
  --smoke-profile           Run visualizer --profile-only on bounded CSV subsets
  --cargo <path>            Cargo executable for --smoke-profile (default: cargo)
  --help                    Show this help
`);
}

function parseArgs(argv) {
  const args = {
    manifest: ".github/datasets/generic-datasets.json",
    cacheDir: ".cache/generic-datasets",
    workDir: "target/generic-datasets",
    schemaOnly: false,
    smokeProfile: false,
    cargo: "cargo",
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      i += 1;
      if (i >= argv.length) throw new Error(`missing value for ${arg}`);
      return argv[i];
    };
    switch (arg) {
      case "--manifest":
        args.manifest = next();
        break;
      case "--cache-dir":
        args.cacheDir = next();
        break;
      case "--work-dir":
        args.workDir = next();
        break;
      case "--schema-only":
        args.schemaOnly = true;
        break;
      case "--smoke-profile":
        args.smokeProfile = true;
        break;
      case "--cargo":
        args.cargo = next();
        break;
      case "--help":
      case "-h":
        usage();
        process.exit(0);
      default:
        throw new Error(`unknown argument: ${arg}`);
    }
  }

  args.cacheDir = path.resolve(ROOT, args.cacheDir);
  args.workDir = path.resolve(ROOT, args.workDir);
  return args;
}

async function loadJson(ref) {
  if (/^https?:\/\//.test(ref)) {
    const response = await fetch(ref);
    if (!response.ok) throw new Error(`GET ${ref} failed with HTTP ${response.status}`);
    return { value: await response.json(), source: ref };
  }
  const filePath = path.resolve(ROOT, ref);
  return { value: JSON.parse(await readFile(filePath, "utf8")), source: filePath };
}

function validateManifest(manifest) {
  const errors = [];
  const seenDatasetIds = new Set();

  if (!isObject(manifest)) {
    throw new Error("manifest must be a JSON object");
  }
  if (manifest.schema_version !== 1) {
    errors.push("schema_version must be 1");
  }
  if (!Array.isArray(manifest.datasets)) {
    errors.push("datasets must be an array");
  }

  for (const [datasetIndex, dataset] of (manifest.datasets || []).entries()) {
    const label = `datasets[${datasetIndex}]`;
    if (!isObject(dataset)) {
      errors.push(`${label} must be an object`);
      continue;
    }
    if (!/^[a-z0-9][a-z0-9._-]*$/.test(dataset.id || "")) {
      errors.push(`${label}.id must match ^[a-z0-9][a-z0-9._-]*$`);
    }
    if (typeof dataset.version !== "string" || dataset.version.length === 0) {
      errors.push(`${label}.version must be a non-empty string`);
    }
    const datasetKey = `${dataset.id}@${dataset.version}`;
    if (seenDatasetIds.has(datasetKey)) {
      errors.push(`${label} duplicates ${datasetKey}`);
    }
    seenDatasetIds.add(datasetKey);

    const replayDir = dataset.replay_dir ?? ".";
    if (!isSafeRelativePath(replayDir)) {
      errors.push(`${label}.replay_dir must be a safe relative path`);
    }
    if (!Array.isArray(dataset.files) || dataset.files.length < 2) {
      errors.push(`${label}.files must contain at least imu.csv and gnss.csv`);
      continue;
    }

    const seenPaths = new Set();
    for (const [fileIndex, file] of dataset.files.entries()) {
      const fileLabel = `${label}.files[${fileIndex}]`;
      if (!isObject(file)) {
        errors.push(`${fileLabel} must be an object`);
        continue;
      }
      if (!isSafeRelativePath(file.path || "")) {
        errors.push(`${fileLabel}.path must be a safe relative path`);
      }
      if (seenPaths.has(file.path)) {
        errors.push(`${fileLabel}.path duplicates ${file.path}`);
      }
      seenPaths.add(file.path);
      if (typeof file.url !== "string" || file.url.length === 0) {
        errors.push(`${fileLabel}.url must be a non-empty HTTP(S) URL or relative path`);
      } else if (!/^https?:\/\//.test(file.url) && path.posix.isAbsolute(file.url.replaceAll("\\", "/"))) {
        errors.push(`${fileLabel}.url must be HTTP(S) or a relative path`);
      }
      if (!/^[a-fA-F0-9]{64}$/.test(file.sha256 || "")) {
        errors.push(`${fileLabel}.sha256 must be a 64-character hex digest`);
      }
      if (file.bytes !== undefined && (!Number.isInteger(file.bytes) || file.bytes <= 0)) {
        errors.push(`${fileLabel}.bytes must be a positive integer when present`);
      }
    }

    const replayFiles = new Set(dataset.files.map((file) => normalizeManifestPath(file.path)));
    const imuPath = normalizeManifestPath(path.posix.join(replayDir.replaceAll("\\", "/"), "imu.csv"));
    const gnssPath = normalizeManifestPath(path.posix.join(replayDir.replaceAll("\\", "/"), "gnss.csv"));
    if (!replayFiles.has(imuPath) && !replayFiles.has(`${imuPath}.gz`)) {
      errors.push(`${label}.files is missing ${imuPath} or ${imuPath}.gz`);
    }
    if (!replayFiles.has(gnssPath) && !replayFiles.has(`${gnssPath}.gz`)) {
      errors.push(`${label}.files is missing ${gnssPath} or ${gnssPath}.gz`);
    }

    if (dataset.smoke !== undefined) {
      if (!isObject(dataset.smoke)) {
        errors.push(`${label}.smoke must be an object`);
      } else {
        for (const key of ["max_imu_rows", "max_gnss_rows"]) {
          if (
            dataset.smoke[key] !== undefined &&
            (!Number.isInteger(dataset.smoke[key]) || dataset.smoke[key] <= 0)
          ) {
            errors.push(`${label}.smoke.${key} must be a positive integer`);
          }
        }
        if (
          dataset.smoke.misalignment !== undefined &&
          !["internal", "external", "ref"].includes(dataset.smoke.misalignment)
        ) {
          errors.push(`${label}.smoke.misalignment must be internal, external, or ref`);
        }
      }
    }
  }

  if (errors.length) {
    throw new Error(`manifest validation failed:\n${errors.map((error) => `  - ${error}`).join("\n")}`);
  }
}

async function prepareDataset(dataset, args, manifestSource) {
  const datasetDir = path.join(args.workDir, "datasets", `${dataset.id}-${dataset.version}`);
  await rm(datasetDir, { recursive: true, force: true });
  await mkdir(datasetDir, { recursive: true });

  for (const file of dataset.files) {
    const cached = await ensureCached(file, args.cacheDir, manifestSource);
    const output = path.join(datasetDir, file.path);
    await mkdir(path.dirname(output), { recursive: true });
    await copyFile(cached, output);
  }

  const replayDir = path.join(datasetDir, dataset.replay_dir ?? ".");
  await validateReplayCsvs(replayDir, `${dataset.id}@${dataset.version}`);
  return { datasetDir, replayDir };
}

async function ensureCached(file, cacheDir, manifestSource) {
  await mkdir(cacheDir, { recursive: true });
  const cached = path.join(cacheDir, `${file.sha256.toLowerCase()}-${path.basename(file.path)}`);
  if (existsSync(cached)) {
    await verifyFile(cached, file);
    console.log(`cache hit ${file.path}`);
    return cached;
  }

  console.log(`fetch ${file.url}`);
  const body = await readManifestFileRef(file.url, manifestSource);
  const actual = sha256Buffer(body);
  if (actual !== file.sha256.toLowerCase()) {
    throw new Error(`${file.path} checksum mismatch: expected ${file.sha256}, got ${actual}`);
  }
  if (file.bytes !== undefined && body.length !== file.bytes) {
    throw new Error(`${file.path} byte count mismatch: expected ${file.bytes}, got ${body.length}`);
  }
  const temp = `${cached}.${process.pid}.tmp`;
  await writeFile(temp, body);
  await rename(temp, cached);
  return cached;
}

async function readManifestFileRef(ref, manifestSource) {
  if (/^https?:\/\//.test(ref)) {
    const response = await fetch(ref);
    if (!response.ok) throw new Error(`GET ${ref} failed with HTTP ${response.status}`);
    return Buffer.from(await response.arrayBuffer());
  }
  if (/^https?:\/\//.test(manifestSource)) {
    const url = new URL(ref, manifestSource).toString();
    const response = await fetch(url);
    if (!response.ok) throw new Error(`GET ${url} failed with HTTP ${response.status}`);
    return Buffer.from(await response.arrayBuffer());
  }
  const baseDir = path.dirname(manifestSource);
  const filePath = path.resolve(baseDir, ref);
  const relative = path.relative(ROOT, filePath);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`relative dataset path escapes repository: ${ref}`);
  }
  return readFile(filePath);
}

async function verifyFile(filePath, expected) {
  const actual = await sha256File(filePath);
  if (actual !== expected.sha256.toLowerCase()) {
    await rm(filePath, { force: true });
    throw new Error(`${filePath} checksum mismatch in cache; removed stale file`);
  }
}

async function validateReplayCsvs(replayDir, label) {
  const imu = await validateCsv(await readReplayCsv(replayDir, "imu.csv"), `${replayDir}/imu.csv`, IMU_HEADER, false);
  const gnss = await validateCsv(
    await readReplayCsv(replayDir, "gnss.csv"),
    `${replayDir}/gnss.csv`,
    GNSS_HEADER,
    true,
  );
  const referenceAttitude = await readOptionalReplayCsv(replayDir, "reference_attitude.csv");
  const referenceMount = await readOptionalReplayCsv(replayDir, "reference_mount.csv");
  const referenceAttitudeRows = referenceAttitude
    ? (await validateCsv(referenceAttitude, `${replayDir}/reference_attitude.csv`, REFERENCE_RPY_HEADER, false)).rows
    : 0;
  const referenceMountRows = referenceMount
    ? (await validateCsv(referenceMount, `${replayDir}/reference_mount.csv`, REFERENCE_RPY_HEADER, false)).rows
    : 0;
  console.log(
    `${label}: imu_rows=${imu.rows} gnss_rows=${gnss.rows} reference_attitude_rows=${referenceAttitudeRows} reference_mount_rows=${referenceMountRows}`,
  );
}

async function readReplayCsv(replayDir, name) {
  const plainPath = path.join(replayDir, name);
  const gzipPath = `${plainPath}.gz`;
  if (existsSync(plainPath)) {
    return readFile(plainPath, "utf8");
  }
  if (existsSync(gzipPath)) {
    const decompressed = await gunzipAsync(await readFile(gzipPath));
    return decompressed.toString("utf8");
  }
  throw new Error(`missing ${plainPath} or ${gzipPath}`);
}

async function readOptionalReplayCsv(replayDir, name) {
  const plainPath = path.join(replayDir, name);
  const gzipPath = `${plainPath}.gz`;
  if (!existsSync(plainPath) && !existsSync(gzipPath)) {
    return null;
  }
  return readReplayCsv(replayDir, name);
}

async function validateCsv(text, filePath, expectedHeader, allowHeadingNan) {
  const lines = text.trimEnd().split(/\r?\n/);
  if (lines[0] !== expectedHeader) {
    throw new Error(`${filePath} header mismatch`);
  }
  if (lines.length < 2) {
    throw new Error(`${filePath} must contain at least one data row`);
  }

  const expectedCols = expectedHeader.split(",").length;
  let prevTime = -Infinity;
  for (let i = 1; i < lines.length; i += 1) {
    if (!lines[i].trim()) continue;
    const cols = lines[i].split(",");
    if (cols.length !== expectedCols) {
      throw new Error(`${filePath}:${i + 1} expected ${expectedCols} columns, got ${cols.length}`);
    }
    for (let col = 0; col < cols.length; col += 1) {
      const value = cols[col].trim();
      if (allowHeadingNan && col === cols.length - 1 && value.toLowerCase() === "nan") {
        continue;
      }
      const parsed = Number(value);
      if (!Number.isFinite(parsed)) {
        throw new Error(`${filePath}:${i + 1} column ${col + 1} is not finite`);
      }
      if (col === 0) {
        if (parsed < prevTime) {
          throw new Error(`${filePath}:${i + 1} timestamp moved backwards`);
        }
        prevTime = parsed;
      }
    }
  }
  return { rows: lines.length - 1 };
}

async function makeSmokeReplay(dataset, replayDir, args) {
  const smoke = dataset.smoke || {};
  const smokeDir = path.join(args.workDir, "smoke", `${dataset.id}-${dataset.version}`);
  await rm(smokeDir, { recursive: true, force: true });
  await mkdir(smokeDir, { recursive: true });
  await writeCsvSubset(path.join(replayDir, "imu.csv"), path.join(smokeDir, "imu.csv"), smoke.max_imu_rows || 1000);
  await writeCsvSubset(
    path.join(replayDir, "gnss.csv"),
    path.join(smokeDir, "gnss.csv"),
    smoke.max_gnss_rows || 200,
  );
  await writeOptionalCsvSubset(
    path.join(replayDir, "reference_attitude.csv"),
    path.join(smokeDir, "reference_attitude.csv"),
    smoke.max_gnss_rows || 200,
  );
  await writeOptionalCsvSubset(
    path.join(replayDir, "reference_mount.csv"),
    path.join(smokeDir, "reference_mount.csv"),
    smoke.max_gnss_rows || 200,
  );
  return smokeDir;
}

async function writeCsvSubset(input, output, maxRows) {
  const text = existsSync(input)
    ? await readFile(input, "utf8")
    : (await gunzipAsync(await readFile(`${input}.gz`))).toString("utf8");
  const lines = text.trimEnd().split(/\r?\n/);
  const subset = lines.slice(0, Math.min(lines.length, maxRows + 1));
  await writeFile(output, `${subset.join("\n")}\n`);
}

async function writeOptionalCsvSubset(input, output, maxRows) {
  if (!existsSync(input) && !existsSync(`${input}.gz`)) {
    return;
  }
  await writeCsvSubset(input, output, maxRows);
}

async function runSmokeProfile(dataset, replayDir, args) {
  if (dataset.smoke?.enabled === false) {
    console.log(`${dataset.id}@${dataset.version}: smoke disabled`);
    return;
  }
  const smokeDir = await makeSmokeReplay(dataset, replayDir, args);
  const misalignment = dataset.smoke?.misalignment || "internal";
  await run(args.cargo, [
    "run",
    "-p",
    "sim",
    "--bin",
    "visualizer",
    "--locked",
    "--",
    "--generic-replay-dir",
    smokeDir,
    "--profile-only",
    "--misalignment",
    misalignment,
  ]);
}

function run(command, args) {
  return new Promise((resolve, reject) => {
    console.log(`run ${command} ${args.join(" ")}`);
    const child = spawn(command, args, { cwd: ROOT, stdio: "inherit" });
    child.on("error", reject);
    child.on("exit", (code, signal) => {
      if (code === 0) resolve();
      else reject(new Error(`${command} exited with code=${code} signal=${signal}`));
    });
  });
}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isSafeRelativePath(value) {
  if (typeof value !== "string" || value.length === 0) return false;
  const normalized = path.posix.normalize(value.replaceAll("\\", "/"));
  return !normalized.startsWith("../") && normalized !== ".." && !path.posix.isAbsolute(normalized);
}

function normalizeManifestPath(value) {
  return path.posix.normalize(value.replaceAll("\\", "/"));
}

function sha256Buffer(buffer) {
  return createHash("sha256").update(buffer).digest("hex");
}

function sha256File(filePath) {
  return new Promise((resolve, reject) => {
    const hash = createHash("sha256");
    createReadStream(filePath)
      .on("data", (chunk) => hash.update(chunk))
      .on("error", reject)
      .on("end", () => resolve(hash.digest("hex")));
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const { value: manifest, source } = await loadJson(args.manifest);
  validateManifest(manifest);
  console.log(`manifest ok: ${source}`);

  if (args.schemaOnly) {
    return;
  }
  if (!manifest.datasets.length) {
    console.log("manifest contains no datasets; download and smoke validation skipped");
    return;
  }

  await mkdir(args.workDir, { recursive: true });
  for (const dataset of manifest.datasets) {
    const prepared = await prepareDataset(dataset, args, source);
    if (args.smokeProfile) {
      await runSmokeProfile(dataset, prepared.replayDir, args);
    }
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
