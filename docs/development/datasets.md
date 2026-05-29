# Dataset Maintenance

This page describes how to package datasets for the hosted browser visualizer or
CI validation. Users who only want to load data should start with
[](../data-and-simulation.md) and [](../tools/visualizer.md).

## Packaging Generic Replay

Use `scripts/package_dataset.py` to create a deterministic package from a
generic replay directory. The package contains `manifest.json`, `imu.csv.gz`,
`gnss.csv.gz`, and any optional reference streams present in the source
directory.

The script packages one dataset directory. It does not update browser or CI
manifest lists by itself.

## Packaging iOS Recordings

First export `.motionfusion` into a generic replay directory:

```bash
cd mobile/ios
python3 scripts/export_motionfusion.py ~/Downloads/session.motionfusion --output-dir /tmp/session-web
```

Then package the exported generic replay with the repo-level dataset packaging
tools. iOS recordings usually do not include reference attitude, position,
mount, or motion streams, so those overlays are absent unless generated
separately.

## Hosted Manifests

The browser dataset picker reads `web/datasets/manifest.json`. The hosted
generic dataset validation job reads `.github/datasets/generic-datasets.json`.
Keep those manifests aligned when adding or replacing hosted datasets.

The current browser manifest contains 32 datasets. See
[](../data/hosted-datasets.md) for the checked-in list.

## Validation

Validate hosted generic datasets with:

```bash
node scripts/validate_generic_datasets.mjs \
  --manifest .github/datasets/generic-datasets.json \
  --cache-dir .cache/generic-datasets \
  --work-dir target/generic-datasets \
  --smoke-profile
```

Browser-facing Pages validation also checks that dataset URLs are safe and that
referenced core files are fetchable from the assembled static artifact.
