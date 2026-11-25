"""Tests for the A5 sidebar multipage navigation router.

All tests use synthetic data or filesystem inspection only; no model downloads.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

APP_DIR = Path(__file__).resolve().parents[1] / "app"
PAGES_DIR = APP_DIR / "pages"


# ---------------------------------------------------------------------------
# 1. Discovery: glob pattern matches numbered pages
# ---------------------------------------------------------------------------


def test_pages_discovered():
    """Router glob discovers all existing NN_*.py pages in app/pages/."""
    page_files = sorted(PAGES_DIR.glob("[0-9][0-9]_*.py"))
    assert len(page_files) >= 4, f"Expected at least 4 pages, found: {[p.name for p in page_files]}"
    names = [p.name for p in page_files]
    assert "01_Lens_Heatmap.py" in names
    assert "02_Probing.py" in names
    assert "03_Compression.py" in names
    assert "04_Crossling.py" in names


def test_pages_sorted_by_number():
    """Pages are sorted numerically so the sidebar order is deterministic."""
    page_files = sorted(PAGES_DIR.glob("[0-9][0-9]_*.py"))
    names = [p.name for p in page_files]
    for i in range(len(names) - 1):
        assert names[i] < names[i + 1], f"Pages not sorted: {names}"


# ---------------------------------------------------------------------------
# 2. app.py router: does not import page modules or tab modules
# ---------------------------------------------------------------------------


def test_app_does_not_import_tabs():
    """app/app.py must not import tabs.* or pages.* at the module level."""
    app_src = (APP_DIR / "app.py").read_text()
    tree = ast.parse(app_src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith(("tabs", "pages")), (
                    f"app.py imports '{module}' — this causes eager loading."
                )
            else:
                for alias in node.names:
                    assert not alias.name.startswith(("tabs", "pages")), (
                        f"app.py imports '{alias.name}' — this causes eager loading."
                    )


def test_app_uses_st_navigation():
    """app/app.py must call st.navigation and st.Page (A5 router API)."""
    app_src = (APP_DIR / "app.py").read_text()
    assert "st.navigation" in app_src, "app.py does not use st.navigation"
    assert "st.Page" in app_src, "app.py does not use st.Page"
    assert "st.tabs" not in app_src, "app.py still uses st.tabs — must use sidebar navigation"


# ---------------------------------------------------------------------------
# 3. Synthetic router: 5th page discovery + helper exclusion + lazy execution
# ---------------------------------------------------------------------------


def test_synthetic_router_discovers_fifth_page_and_executes_lazily(tmp_path):
    """A synthetic 5th page is discovered without editing app.py, helpers are
    excluded, and only the selected page executes."""
    from streamlit.testing.v1 import AppTest

    router_copy = tmp_path / "app.py"
    router_copy.write_text((APP_DIR / "app.py").read_text())

    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    log_file = tmp_path / "executed.txt"

    for idx, title in ((1, "First"), (2, "Second"), (3, "Third"), (4, "Fourth"), (5, "Fifth")):
        (pages_dir / f"0{idx}_{title}.py").write_text(
            "import streamlit as st\n"
            "from pathlib import Path\n"
            f"with Path({str(log_file)!r}).open('a') as f:\n"
            f"    f.write('{idx}\\n')\n"
            f"st.title('{title}')\n"
        )
    # Helper module inside pages/ must not be registered by the [0-9][0-9]_*.py glob.
    (pages_dir / "helper_module.py").write_text(
        f"from pathlib import Path\nPath({str(log_file)!r}).write_text('helper\\n')\n"
    )

    at = AppTest.from_file(str(router_copy))
    at.run()
    assert not at.exception
    assert [t.value for t in at.title] == ["First"]
    executed = log_file.read_text().splitlines()
    assert executed == ["1"], f"Expected lazy execution of page 1 only, got: {executed}"

    # Switch to the synthetic 5th page using st.switch_page via session/page run
    at.switch_page("pages/05_Fifth.py").run()
    assert not at.exception
    assert [t.value for t in at.title] == ["Fifth"]
    executed = log_file.read_text().splitlines()
    assert executed == ["1", "5"], f"Expected only pages 1 and 5 to execute, got: {executed}"


# ---------------------------------------------------------------------------
# 4. Lazy loading: importing probing tab does not load torch
# ---------------------------------------------------------------------------


def test_probing_tab_does_not_load_torch():
    """Importing app/tabs/probing must not cause torch to be imported."""
    torch_keys = [k for k in sys.modules if k == "torch" or k.startswith("torch.")]
    saved = {k: sys.modules.pop(k) for k in torch_keys}

    for key in [k for k in sys.modules if k == "tabs.probing" or k.startswith("tabs.probing.")]:
        sys.modules.pop(key)

    try:
        if str(APP_DIR) not in sys.path:
            sys.path.insert(0, str(APP_DIR))
        import tabs.probing  # noqa: F401

        assert "torch" not in sys.modules, (
            "Importing tabs.probing caused torch to be loaded."
        )
    finally:
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# 5. UI/helper separation: pages have UI body, tabs do not expose render()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "page_name",
    [
        "01_Lens_Heatmap.py",
        "02_Probing.py",
        "03_Compression.py",
        "04_Crossling.py",
    ],
)
def test_page_has_ui_body_and_imports_helpers(page_name: str):
    """Page files must contain direct st.* widget calls and import helpers from tabs."""
    page_src = (PAGES_DIR / page_name).read_text()
    assert "st." in page_src, f"{page_name} has no st.* calls — it must contain the UI body."
    assert "from tabs." in page_src or "import tabs." in page_src, (
        f"{page_name} does not import any helper from app/tabs/."
    )


@pytest.mark.parametrize(
    "tab_name",
    ["lens_heatmap.py", "probing.py", "cka.py", "crossling.py"],
)
def test_tab_helper_has_no_render(tab_name: str):
    """Migrated helper modules under app/tabs/ must not keep a UI render() function."""
    tree = ast.parse((APP_DIR / "tabs" / tab_name).read_text())
    top_defs = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "render" not in top_defs, f"app/tabs/{tab_name} still defines render()."


# ---------------------------------------------------------------------------
# 6. Page-level UI smoke tests (synthetic data / missing artifacts)
# ---------------------------------------------------------------------------


def test_probing_page_synthetic_does_not_load_torch(monkeypatch, tmp_path):
    """Running 02_Probing.py renders curves and CoG without importing torch."""
    from streamlit.testing.v1 import AppTest

    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    from tabs import probing as probing_tab

    probing_dir = tmp_path / "probing"
    probing_dir.mkdir()
    rows = []
    for model, kind in (("bert", "encoder"), ("gpt2", "decoder")):
        for task in ("upos", "control"):
            for layer in range(3):
                rows.append({
                    "model": model, "kind": kind, "lang": "en", "probe": "linear",
                    "train_treebank": "ewt", "eval_treebank": "ewt", "task": task,
                    "init": "pretrained", "pooling": "first", "seed": 42,
                    "layer": layer, "L": 2, "tau": layer / 2,
                    "acc": 0.7 + 0.05 * layer if task == "upos" else 0.4,
                    "n_train": 50, "n_eval": 20, "hparams": "{}",
                })
    pd.DataFrame(rows).to_parquet(probing_dir / "synthetic.parquet")

    summary_path = tmp_path / "cog.parquet"
    pd.DataFrame([
        {
            "model": "bert", "train_treebank": "ewt", "eval_treebank": "ewt",
            "probe": "linear", "init": "pretrained", "pooling": "first",
            "task": "upos", "cog_norm": 0.6,
        },
        {
            "model": "gpt2", "train_treebank": "ewt", "eval_treebank": "ewt",
            "probe": "linear", "init": "pretrained", "pooling": "first",
            "task": "upos", "cog_norm": 0.55,
        },
    ]).to_parquet(summary_path)

    monkeypatch.setattr(probing_tab, "PROBING_DIR", probing_dir)
    monkeypatch.setattr(probing_tab, "SUMMARY_PATH", summary_path)

    torch_keys = [k for k in sys.modules if k == "torch" or k.startswith("torch.")]
    saved = {k: sys.modules.pop(k) for k in torch_keys}
    try:
        page = AppTest.from_file(str(PAGES_DIR / "02_Probing.py"))
        page.run()
        assert not page.exception
        assert len(page.get("plotly_chart")) == 1
        assert len(page.dataframe) == 1
        assert "torch" not in sys.modules
    finally:
        sys.modules.update(saved)


def test_lens_page_empty_text_does_not_load_model(monkeypatch):
    """Empty sentence on 01_Lens_Heatmap.py shows an info message without loading models."""
    from streamlit.testing.v1 import AppTest

    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    from tabs import lens_heatmap

    monkeypatch.setitem(lens_heatmap.DEFAULT_SENTENCES, "gpt2", "")
    monkeypatch.setattr(
        lens_heatmap,
        "load_lens_params",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not load params")),
    )
    monkeypatch.setattr(
        lens_heatmap,
        "load_lens_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not load model")),
    )
    page = AppTest.from_file(str(PAGES_DIR / "01_Lens_Heatmap.py"))
    page.run(timeout=30)
    assert not page.exception
    assert len(page.info) == 1
    assert "Enter a sentence" in page.info[0].value


def test_pages_md_exists():
    """app/PAGES.md must exist and document the page filenames."""
    pages_md = APP_DIR / "PAGES.md"
    assert pages_md.is_file(), "app/PAGES.md is missing."
    content = pages_md.read_text()
    for name in ("01_Lens_Heatmap.py", "02_Probing.py", "03_Compression.py", "04_Crossling.py"):
        assert name in content, f"app/PAGES.md does not mention {name}."


@pytest.mark.parametrize(
    "page_name",
    [
        "01_Lens_Heatmap.py",
        "02_Probing.py",
        "03_Compression.py",
        "04_Crossling.py",
        "05_Linear_vs_MLP.py",
    ],
)
def test_page_is_valid_python(page_name: str):
    """Each page file must parse as valid Python (syntax check)."""
    page_path = PAGES_DIR / page_name
    if not page_path.is_file():
        pytest.skip(f"{page_name} not found.")
    ast.parse(page_path.read_text())
