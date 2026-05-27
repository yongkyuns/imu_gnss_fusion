# CI And Pages Deployment

The CI workflow has three main responsibilities:

- Rust validation on Linux and macOS: format, clippy, workspace build, workspace tests, selected sim binary checks, and `cargo doc`.
- Hosted generic dataset validation: manifest/schema/checksum validation plus smoke profiling.
- GitHub Pages artifact assembly: wasm visualizer build, Sphinx docs build, static artifact validation, and upload.

```{figure} ../_static/diagrams/pages-artifact.svg
:alt: GitHub Pages artifact layout showing visualizer root and docs subdirectory.
:class: framed

Pages deployment preserves the visualizer at `/` and publishes the documentation site under `/docs/`.
```

The Pages artifact keeps the visualizer at the root and copies generated Sphinx HTML to `/docs/`:

```text
target/pages-site/
  index.html
  pkg/
  datasets/
  docs/
```

The generated Pages artifact is not committed. CI validates it before upload so broken wasm paths, missing docs, unsafe dataset paths, or missing static files fail before deployment.
