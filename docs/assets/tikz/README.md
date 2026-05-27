# TikZ Diagrams

These files are the editable sources for selected documentation diagrams. The
generated PNGs are committed under `docs/_static/diagrams/` so the Sphinx build
does not require a LaTeX installation in CI.

Regenerate one diagram locally with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory target/tikz docs/assets/tikz/overall-architecture.tex
magick -density 220 target/tikz/overall-architecture.pdf -background white -alpha remove -alpha off docs/_static/diagrams/overall-architecture.png
```
