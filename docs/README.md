# Documentation Source

This directory is the Sphinx source tree for the public documentation site:

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs target/docs-html
```

The built site is deployed under `/docs/` in the GitHub Pages artifact. The Pages root remains the wasm visualizer from `web/index.html`.

Start from [index.md](index.md) when editing documentation. Algorithm and formulation notes should be maintained as normal Sphinx pages so the public site stays readable without separate document artifacts.
