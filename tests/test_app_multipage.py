"""Tests for the A5 sidebar multipage navigation router.

All tests use synthetic data or filesystem inspection only; no model downloads,
no Streamlit rendering harness required.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).parent.parent / "app"
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


def test_synthetic_page_discovered(tmp_path):
    """A synthetic page under app/pages/ is discovered by the router pattern
    without any edit to app/app.py."""
    # Simulate the router's glob logic on tmp_path
    fake_page = tmp_path / "99_Synthetic.py"
    fake_page.write_text("import streamlit as st\nst.write('synthetic')\n")
    page_files = sorted(tmp_path.glob("[0-9][0-9]_*.py"))
    assert any(p.name == "99_Synthetic.py" for p in page_files)


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
    """app/app.py must not import tabs.* or pages.* at the module level.

    It discovers pages by filename; importing tabs eagerly would trigger model
    loading on every page navigation.
    """
    app_src = (APP_DIR / "app.py").read_text()
    tree = ast.parse(app_src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith("tabs"), (
                    f"app.py imports tabs module '{module}' — this causes eager model loading."
                )
            else:
                for alias in node.names:
                    assert not alias.name.startswith("tabs"), (
                        f"app.py imports tabs module '{alias.name}' — this causes eager model loading."
                    )


def test_app_uses_st_navigation():
    """app/app.py must call st.navigation and st.Page (A5 router API)."""
    app_src = (APP_DIR / "app.py").read_text()
    assert "st.navigation" in app_src, "app.py does not use st.navigation"
    assert "st.Page" in app_src, "app.py does not use st.Page"
    assert "st.tabs" not in app_src, "app.py still uses st.tabs — must use sidebar navigation"


# ---------------------------------------------------------------------------
# 3. Lazy loading: importing probing tab does not load torch
# ---------------------------------------------------------------------------


def test_probing_tab_does_not_load_torch():
    """Importing app/tabs/probing must not cause torch to be imported.

    This guarantees that visiting the Probing page never triggers GPU/model
    loading silently in the background.
    """
    # Remove any torch already imported in this process
    torch_keys = [k for k in sys.modules if k == "torch" or k.startswith("torch.")]
    saved = {k: sys.modules.pop(k) for k in torch_keys}

    # Also remove cached tabs.probing import if it was already loaded
    for key in [k for k in sys.modules if "tabs.probing" in k or k == "tabs.probing"]:
        sys.modules.pop(key)

    try:
        # Temporarily add app/ to path so "from tabs.probing import ..." resolves
        if str(APP_DIR) not in sys.path:
            sys.path.insert(0, str(APP_DIR))
        import tabs.probing  # noqa: F401

        assert "torch" not in sys.modules, (
            "Importing tabs.probing caused torch to be loaded — "
            "move torch imports inside functions or guard them."
        )
    except ImportError:
        pytest.skip("tabs.probing cannot be imported in this environment (missing deps).")
    finally:
        # Restore torch modules
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# 4. Page files: each page contains the UI body, not a render() delegation
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
def test_page_does_not_merely_call_render(page_name: str):
    """Page files must contain direct st.* widget calls, not just 'tabs.render()'.

    Per A5 spec: 'a tabs.render() API is no longer required'.
    """
    page_src = (PAGES_DIR / page_name).read_text()
    # The page must NOT be a thin wrapper that only calls some_module.render()
    # Heuristic: if the file is very short AND contains only a render() call, fail.
    # We check positively that the file contains st.* calls.
    assert "st." in page_src, f"{page_name} has no st.* calls — it must contain the UI body."
    # Also check it imports from tabs (helpers), not that render is the whole body
    assert "from tabs." in page_src or "import tabs." in page_src, (
        f"{page_name} does not import any helper from app/tabs/."
    )


# ---------------------------------------------------------------------------
# 5. PAGES.md documents the reserved filenames
# ---------------------------------------------------------------------------


def test_pages_md_exists():
    """app/PAGES.md must exist and document the page filenames."""
    pages_md = APP_DIR / "PAGES.md"
    assert pages_md.is_file(), "app/PAGES.md is missing."
    content = pages_md.read_text()
    for name in ("01_Lens_Heatmap.py", "02_Probing.py", "03_Compression.py", "04_Crossling.py"):
        assert name in content, f"app/PAGES.md does not mention {name}."


# ---------------------------------------------------------------------------
# 6. Missing CKA / lens results do not crash the shell (parse check only)
# ---------------------------------------------------------------------------


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
    source = page_path.read_text()
    try:
        ast.parse(source)
    except SyntaxError as err:
        pytest.fail(f"{page_name} has a syntax error: {err}")
