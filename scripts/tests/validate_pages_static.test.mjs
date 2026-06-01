import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import test from "node:test";

const execFileAsync = promisify(execFile);
const ROOT = path.resolve(import.meta.dirname, "../..");
const SCRIPT = path.join(ROOT, "scripts", "validate_pages_static.mjs");

test("validate_pages_static accepts rendered docs math assets", async () => {
  const fixture = await createFixture();

  await runValidator(fixture);
});

test("validate_pages_static rejects LaTeX inline delimiters in docs source", async () => {
  const fixture = await createFixture({
    sourceMarkdown: "Bad inline symbol: \\(q_{bv}\\).\n",
  });

  const result = await runValidator(fixture, { expectFailure: true });

  assert.match(result.stderr, /raw LaTeX math delimiters/);
  assert.match(result.stderr, /bad\.md:1/);
});

test("validate_pages_static rejects unrendered inline math in built docs HTML", async () => {
  const fixture = await createFixture({
    extraHtml: '<html><body><main><p>Bad inline symbol: (q_{bv}).</p></main></body></html>',
  });

  const result = await runValidator(fixture, { expectFailure: true });

  assert.match(result.stderr, /unrendered inline math/);
  assert.match(result.stderr, /bad\.html/);
});

async function createFixture(options = {}) {
  const root = await mkdtemp(path.join(tmpdir(), "pages-validator-"));
  const siteDir = path.join(root, "site");
  const docsSourceDir = path.join(root, "docs-source");
  await mkdir(path.join(siteDir, "docs", "_static"), { recursive: true });
  await mkdir(docsSourceDir, { recursive: true });

  await writeFile(
    path.join(siteDir, "index.html"),
    '<canvas id="visualizer_canvas"></canvas><script type="module" src="./pkg/visualizer.js"></script>',
  );
  await writeFile(
    path.join(siteDir, "docs", "index.html"),
    [
      "<html><head>",
      '<script src="_static/math-overflow.js"></script>',
      "</head><body>",
      "<main>IMU/GNSS Fusion ",
      '<span class="math notranslate nohighlight">\\(q_{bv}\\)</span>',
      "</main></body></html>",
    ].join(""),
  );
  await writeStaticAssets(siteDir);
  await writeFile(
    path.join(docsSourceDir, "good.md"),
    "Good inline symbol: $q_{bv}$.\n\n```text\n\\(allowed_in_fence\\)\n```\n",
  );
  if (options.sourceMarkdown) {
    await writeFile(path.join(docsSourceDir, "bad.md"), options.sourceMarkdown);
  }
  if (options.extraHtml) {
    await writeFile(path.join(siteDir, "docs", "bad.html"), options.extraHtml);
  }

  return { siteDir, docsSourceDir };
}

async function writeStaticAssets(siteDir) {
  const staticDir = path.join(siteDir, "docs", "_static");
  const png = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"></svg>\n';
  await mkdir(path.join(staticDir, "diagrams"), { recursive: true });
  await mkdir(path.join(staticDir, "screenshots"), { recursive: true });
  await writeFile(path.join(staticDir, "custom.css"), ".math-no-scroll { overflow-x: clip; }\n");
  await writeFile(
    path.join(staticDir, "math-overflow.js"),
    "function updateMathOverflow() { document.body.classList.add('math-no-scroll'); }\n",
  );
  await writeFile(path.join(staticDir, "logo.png"), png);
  await writeFile(path.join(staticDir, "titlebar.png"), png);
  await writeFile(path.join(staticDir, "diagrams", "estimator-runtime-orthogonal.svg"), svg);
  await writeFile(path.join(staticDir, "diagrams", "overall-runtime-architecture.svg"), svg);
  await writeFile(path.join(staticDir, "screenshots", "web-visualizer-overview.png"), png);
}

async function runValidator(fixture, options = {}) {
  const args = [
    SCRIPT,
    "--site-dir",
    fixture.siteDir,
    "--docs-source-dir",
    fixture.docsSourceDir,
    "--require-docs",
  ];
  try {
    const result = await execFileAsync(process.execPath, args, {
      cwd: ROOT,
      encoding: "utf8",
    });
    if (options.expectFailure) {
      assert.fail("validator unexpectedly succeeded");
    }
    return result;
  } catch (error) {
    if (!options.expectFailure) {
      throw error;
    }
    return {
      stdout: error.stdout ?? "",
      stderr: error.stderr ?? "",
      code: error.code,
    };
  }
}
