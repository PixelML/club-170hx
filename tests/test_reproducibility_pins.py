from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validate_recipe_notebooks", REPO_ROOT / "scripts/validate_recipe_notebooks.py")
assert spec and spec.loader
validator = importlib.util.module_from_spec(spec)
sys.modules["validate_recipe_notebooks"] = validator
spec.loader.exec_module(validator)

FULL = "sha256:" + "a" * 64


def notebook(text: str) -> str:
    return json.dumps({"cells": [{"cell_type": "markdown", "source": [text]}], "nbformat": 4})


class ReproducibilityPinTests(unittest.TestCase):
    def check(self, name: str, text: str) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notebooks" / name
            path.parent.mkdir()
            path.write_text(notebook(text), encoding="utf-8")
            return validator.pin_errors(path)

    def test_full_digest_passes(self) -> None:
        self.assertEqual(self.check("2026-10-02-x.ipynb", f"image@{FULL} built from pinned commits"), [])

    def test_truncated_digest_fails(self) -> None:
        for text in ("image@sha256:cc26c8abb639…", "image@sha256:cc26c8abb639...", "image@sha256:cc26c8abb639 next"):
            self.assertTrue(self.check("2026-10-02-x.ipynb", text), text)

    def test_unpublished_image_fails(self) -> None:
        self.assertTrue(self.check("2026-10-02-x.ipynb", "pixelml/foo:tag (built locally, not published)"))

    def test_dated_documents_are_checked_and_templates_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notebooks").mkdir()
            for name in ("2026-09-30-old.ipynb", "2026-10-02-new.ipynb", "TEMPLATE.ipynb"):
                (root / "notebooks" / name).write_text(notebook("x"), encoding="utf-8")
            checked = [p.name for p in validator.pinned_documents(root)]
            self.assertEqual(checked, ["2026-09-30-old.ipynb", "2026-10-02-new.ipynb"])


if __name__ == "__main__":
    unittest.main()
