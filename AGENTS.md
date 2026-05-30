# Agent Notes

For this project, persistent project knowledge should be maintained using the
LLM-wiki structure at:

- `/Users/ykshin/Documents/Obsidian Vault/Agent/imu_gnss_fusion/index.md`

Before starting project behavior work, read the wiki index and the
relevant linked pages. When a finding should carry across future work:

- update the relevant maintained wiki page under
  `/Users/ykshin/Documents/Obsidian Vault/Agent/imu_gnss_fusion/pages/`
- append a short entry to
  `/Users/ykshin/Documents/Obsidian Vault/Agent/imu_gnss_fusion/log.md`
- preserve concrete evidence: command lines, dataset names, branch/commit IDs,
  metrics, and artifact paths
- mark uncertain items as hypotheses rather than facts

An archived raw note remains available:

- `/Users/ykshin/Documents/Obsidian Vault/Agent/imu_gnss_fusion.md`

Prefer promoting durable synthesis into the wiki instead of continuing to append
large unstructured sections to the archived note.

## Documentation Tone

The public Sphinx documentation should read like technical reference material,
similar in tone to OpenIMU, PX4 EKF2, and Open Aided Navigation documentation.
When editing docs:

- keep wording concise, neutral, and formulation-focused;
- avoid promotional or competitive language;
- describe state definitions, measurement models, assumptions, limitations,
  configuration, and implementation differences directly;
- use accurate estimator terminology, especially around EKF, AHRS, INS, NHC,
  mount states, attitude, bias, observability, covariance, and priors;
- compare prior work by domain, sensor suite, state representation,
  measurement set, and modeling assumptions rather than by ranking projects;
- state limitations plainly, including ambiguity or observability limits, without
  overstating what the implementation can infer from IMU/GNSS alone.

## Web Visualizer GIF Capture

Use repo-local capture artifacts for README/docs GIFs. The workflow is
project-specific because the visualizer is an egui/WebGPU canvas and the docs
site copies assets from `docs/_static/`.

- Serve the current Pages artifact from the repo root, usually with:
  `python3 -m http.server 8099 --bind 0.0.0.0 --directory target/pages-site`.
  The raw web visualizer is commonly served on `8080` from `web/`, but README
  and docs captures should prefer `target/pages-site` when verifying final
  documentation paths.
- Install Playwright only in ignored scratch space when needed:
  `npm --prefix target/playwright-capture install playwright@1.60.0`.
- Launch Chromium headed with WebGPU flags. Ordinary headless Chromium may fail
  with `No suitable graphics adapter found`.
  Use `headless: false` and args
  `["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"]`.
- Capture frames into `output/playwright/<name>-frames/`. Keep reusable capture
  scripts under `target/*.mjs` unless the script is promoted into the repo.
- Prefer URL-driven setup for stable examples, for example
  `http://127.0.0.1:8080/?theme=dark&dataset=ios-mixed-drive-reverse-parking-001`.
  The `dataset` query auto-loads hosted manifest entries after the manifest is
  fetched.
- For tab/page switching inside the egui canvas, keyboard traversal is more
  reliable than synthetic mouse clicks. Focus `#visualizer_canvas`, press `Tab`
  until the page tabs are focused, then press `Enter` and advance with `Tab`.
  This avoids coordinate mismatch issues in headed WebGPU captures.
- Encode GIFs with ffmpeg palette generation to keep assets small:
  `ffmpeg -y -framerate 8 -i output/playwright/<frames>/frame_%04d.png -vf 'fps=8,scale=960:-1:flags=lanczos,palettegen=max_colors=128' output/playwright/<name>-palette.png`
  followed by
  `ffmpeg -y -framerate 8 -i output/playwright/<frames>/frame_%04d.png -i output/playwright/<name>-palette.png -lavfi 'fps=8,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5' -loop 0 docs/_static/gifs/<name>.gif`.
- After adding GIFs to docs, run:
  `.venv-docs/bin/sphinx-build -W --keep-going -b html docs target/docs-html`,
  sync to the Pages artifact if the local server is active, and run
  `node scripts/validate_pages_static.mjs --site-dir target/pages-site --require-wasm --require-docs`.
