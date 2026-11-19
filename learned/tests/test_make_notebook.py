"""Tests for learned/make_notebook.py: notebook 03's structure (no data needed)."""
import ast
import importlib.util
import re
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("make_learned_notebook", ROOT / "learned/make_notebook.py")
mln = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mln)

TITLES = ["Settings", "Lab measurements and noise floor", "Pieces from notebook 02", "Why this is hard",
          "Yardsticks", "DL-0", "Crops", "DINOv2 features", "DL-1", "DL-2", "Results",
          "Leave one clip out (diagnostic)", "Save everything"]


@pytest.fixture(scope="module")
def nb():
    return mln.build()


def test_steps_present(nb):
    heads = [line for c in nb.cells if c.cell_type == "markdown" for line in c.source.splitlines()
             if line.startswith("## ")]
    steps = [h for h in heads if h.startswith("## Step")]
    assert steps == [f"## Step {i}: {t}" for i, t in enumerate(TITLES, 1)]
    assert heads[-1] == "## Notes and limits"


def test_outputs_cleared(nb):
    for c in nb.cells:
        if c.cell_type == "code":
            assert c.outputs == [] and c.execution_count is None


def test_logic_lives_in_src(nb):
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        tree = ast.parse(re.sub(r"^%.*$", "", c.source, flags=re.M))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert node.end_lineno - node.lineno + 1 <= 15, node.name


def test_settings_cell(nb):
    code = [c for c in nb.cells if c.cell_type == "code"]
    settings = next(c for c in code if "RUN_DL2" in c.source)
    values = {}
    for node in ast.parse(settings.source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            values[node.targets[0].id] = node.value
    for name in ("RESULTS_DIR", "VIDEOS_DIR", "LAB_SHEET"):
        assert name in values
    lit = {k: ast.literal_eval(v) for k, v in values.items()
           if k in ("KEYS", "VERSIONS", "RUN_DL2", "RUN_CLIP_DIAGNOSTIC", "RUN_PERMUTATION", "DEVICE")}
    assert lit == {"KEYS": ("red", "blue"), "VERSIONS": ("outline", "colour"), "RUN_DL2": True, "RUN_CLIP_DIAGNOSTIC": True,
                   "RUN_PERMUTATION": False, "DEVICE": "auto"}


def test_written_notebook_matches_build(tmp_path):
    path = tmp_path / "nb.ipynb"
    mln.main(["--target", str(path)])
    assert nbformat.read(path, as_version=4).cells == mln.build().cells


def test_committed_notebook_is_the_build():
    path = ROOT / "notebooks/03_learned_pathway.ipynb"
    assert nbformat.read(path, as_version=4).cells == mln.build().cells
