# Testing

Run the workspace checks:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo build --workspace --locked
cargo test --workspace --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
```

Validate hosted generic datasets:

```bash
node scripts/validate_generic_datasets.mjs \
  --manifest .github/datasets/generic-datasets.json \
  --cache-dir .cache/generic-datasets \
  --work-dir target/generic-datasets \
  --smoke-profile
```

Build and validate the Pages artifact locally:

```bash
python -m pip install -r docs/requirements.txt
npm ci
npm run render:elk-diagrams
sphinx-build -W --keep-going -b html docs target/docs-html

rm -rf target/pages-site
mkdir -p target/pages-site
rsync -a \
  --exclude local-config.js \
  --exclude pkg/ \
  --exclude docs/ \
  web/ target/pages-site/

cargo build -p sim --bin visualizer --release --target wasm32-unknown-unknown --locked
wasm-bindgen --target web --out-dir target/pages-site/pkg \
  target/wasm32-unknown-unknown/release/visualizer.wasm

mkdir -p target/pages-site/docs
rsync -a target/docs-html/ target/pages-site/docs/
touch target/pages-site/.nojekyll
```

CI assembles the full static artifact under `target/pages-site` and validates it with:

```bash
node scripts/validate_pages_static.mjs \
  --site-dir target/pages-site \
  --require-wasm \
  --require-docs
```
