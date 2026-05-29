#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONF_PATH = ROOT / "docs" / "conf.py"


class DocsConfigTests(unittest.TestCase):
    def test_myst_math_and_static_assets_are_configured(self) -> None:
        spec = importlib.util.spec_from_file_location("docs_conf", CONF_PATH)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertIn("myst_parser", module.extensions)
        self.assertIn("dollarmath", module.myst_enable_extensions)
        self.assertIn("amsmath", module.myst_enable_extensions)
        self.assertIn("_static", module.html_static_path)
        self.assertIn("custom.css", module.html_css_files)
        self.assertIn("math-overflow.js", module.html_js_files)


if __name__ == "__main__":
    unittest.main()
