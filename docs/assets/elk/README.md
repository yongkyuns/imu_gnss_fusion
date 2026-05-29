---
orphan: true
---

# ELK Diagram Sources

These JSON files are source specs for diagrams rendered with the Eclipse Layout
Kernel through `elkjs`. ELK is the canonical source pipeline for generated docs
flowcharts.

Use per-diagram profiles instead of separate layout engines:

- `airy`: spread-out explanatory diagrams. Uses spline routing and relaxed
  ordering so edges can behave more like a clean Graphviz flow.
- `structured`: normal left-to-right process diagrams where model order should
  be mostly preserved.
- `dense`: compact lifecycle/state/decision diagrams. Uses orthogonal routing,
  explicit ports, and rounded rendered bends.

Regenerate the committed SVG outputs with:

```bash
npm install
npm run render:elk-diagrams
```

Generated SVG diagrams are committed under `docs/_static/diagrams/` so Sphinx
and GitHub Pages builds do not require ELK or Node package installation.

Airy diagrams may add numeric `lane` hints to keep related nodes horizontally
aligned. The renderer still uses ELK for layer placement, then aligns nodes
within each lane and draws one smooth connector per visible edge so architecture
diagrams do not accumulate unnecessary routing breakpoints.

When a cross-lane edge would pass through another block, add an explicit edge
`route`, such as `below`, to reserve a clear channel around the intermediate
nodes.
