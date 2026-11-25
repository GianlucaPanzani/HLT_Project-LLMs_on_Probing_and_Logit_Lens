# Streamlit pages and helpers

This document defines the target organization for A5 and the page issues. It
does not imply that every reserved page or the sidebar router is already
implemented. Ownership and scheduling are also recorded in
[docs/TODO.md](../docs/TODO.md#74-interfaces-and-conflict-avoidance).

## UI and helper separation

- `app/pages/NN_*.py` contains the page's UI body: Streamlit widgets, layout,
  captions, displayed errors, charts, tables and download buttons. It imports
  and calls supporting functions from `app/tabs/`.
- `app/tabs/` contains reusable loading, validation, alignment, computation and
  data-preparation functions. These functions return data, figures or status
  information, or raise explicit errors; they do not create widgets or render
  the page. Importing a helper module must not trigger UI rendering, model
  loading or data processing.
- Reusable scientific plotting stays in `src/viz/`; the page displays the
  resulting figures with Streamlit.
- When migrating an existing page, move the UI body out of its `render()`
  function into the page file. A page must not merely delegate its UI to
  `tabs.render()`. Preserve scientific behavior and avoid rendering the UI
  twice.
- Keep widget keys unique across pages. Write widget calls over multiple lines,
  with named arguments and a trailing comma.

## Navigation and loading

The single launch command remains:

```bash
PYTHONPATH=src streamlit run app/app.py
```

Under A5, `app/app.py` owns global page configuration and discovers sorted
`app/pages/[0-9][0-9]_*.py` filenames. It registers file paths with `st.Page` and
`st.navigation(position="sidebar")`, then calls `.run()` only on the selected
page. Discovery must not import or execute every page. Adding a page requires
no edit to a shared registration list; helper modules stay outside `pages/`.

Only the selected page invokes its data or model helpers. Cache live models
where needed and run inference only after the relevant user action. Result-only
pages must not load models. Missing artifacts should produce a message in the
page while keeping navigation usable.

## Page ownership

Filenames and titles below are reserved; they are not an availability inventory.

| Filename under `app/pages/` | Sidebar title | Owner |
| :-- | :-- | :-- |
| `01_Lens_Heatmap.py` | Lens Heatmap | Matteo |
| `02_Probing.py` | Probing | Gianluca |
| `03_Compression.py` | Compression | Filippo |
| `04_Crossling.py` | Crossling | Arcangelo |
| `05_Linear_vs_MLP.py` | Linear vs MLP | Gianluca |
| `06_Pooling.py` | Pooling | Gianluca |
| `07_UPOS_and_Tokens.py` | UPOS and Tokens | Matteo |
| `08_mGPT_UPOS_and_Tokens.py` | mGPT UPOS and Tokens | Arcangelo |
| `09_Prediction_Lab.py` | Prediction Lab | Arcangelo; optional A6 |

For A6, UI composition belongs in `09_Prediction_Lab.py`. Reusable orchestration
belongs in `app/tabs/prediction_lab.py`; owner-specific panel helpers belong in
`app/tabs/lab_panels/`. Those helpers return data, figures or status information
for the page to display.

## Verification

Test helpers with small synthetic inputs, without model downloads or UI
rendering. Test widgets and displayed messages by executing the page file with
Streamlit's test harness. A5 tests filename discovery and execution of only the
selected page, including a synthetic additional page and missing-artifact cases.
Moving UI code must preserve the existing helper behavior and result formats.
