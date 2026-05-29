from __future__ import annotations

from pathlib import Path

project = "IMU/GNSS Fusion"
author = "IMU/GNSS Fusion contributors"
copyright = "2026, IMU/GNSS Fusion contributors"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
]

source_suffix = {
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
]

html_theme = "furo"
html_title = "IMU/GNSS Fusion"
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["custom.css"]
html_baseurl = "https://yongkyuns.github.io/imu_gnss_fusion/docs/"
html_theme_options = {
    "source_repository": "https://github.com/yongkyuns/imu_gnss_fusion/",
    "source_branch": "main",
    "source_directory": "docs/",
}

suppress_warnings = [
    "myst.header",
]

nitpicky = False

root = Path(__file__).resolve().parents[1]
