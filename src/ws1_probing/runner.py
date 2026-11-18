import numpy as np
import pandas as pd

from lib.control_tasks import control_labels, load_control
from lib.data import UPOS_TAGS
from lib.registry import get_model_spec, get_treebank_spec
from lib.features import load_layer, load_meta
from lib.word_table import load_word_table
from probes import LinearProbe, MLPProbe


def _prepare_split(
    model: str,
    init: str,
    pooling: str,
    treebank: str,
    split: str,
    tasks: tuple[str, ...],
    control: dict | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return valid row positions and aligned integer labels."""
    table = load_word_table(treebank, split)
    meta = load_meta(model, init, pooling, treebank, split)
    features = load_layer(model, init, pooling, treebank, split, 0)

    if features.ndim != 2 or len(features) != len(table):
        raise ValueError(f"Features and word table do not align for {treebank}/{split}.")

    # Capped extractions cover the first max_rows word-table rows.
    max_rows = meta.get("max_rows")
    if max_rows is not None:
        table = table.iloc[:max_rows]

    tag_to_id = {tag: index for index, tag in enumerate(UPOS_TAGS)}
    valid = ~np.isnan(features).any(axis=1)
    encoded_labels = {}

    for task in tasks:
        if task == "upos":
            labels = table["upos"]
        elif task == "control":
            if control is None:
                raise ValueError("Control mapping is required.")
            labels = control_labels(table, control)
        else:
            raise ValueError(f"Unsupported task: {task!r}.")

        unknown = labels.notna() & ~labels.isin(UPOS_TAGS)
        if unknown.any():
            raise ValueError(f"Unknown labels for {task} in {treebank}/{split}.")

        valid &= labels.notna().to_numpy()
        encoded_labels[task] = labels.map(tag_to_id)

    rows = np.flatnonzero(valid)
    if rows.size == 0:
        raise ValueError(f"No valid rows for {treebank}/{split}.")

    targets = {
        task: labels.iloc[rows].to_numpy(dtype=np.int64)
        for task, labels in encoded_labels.items()
    }
    return rows, targets


def run_probing(
        model: str,
        train_treebank: str,
        *,
        probe: str,
        init: str,
        pooling: str,
        tasks: tuple[str, ...] = ("upos", "control"),
        eval_treebanks: tuple[str, ...] | None = None,
        layers: list[int] | None = None,
        max_train_rows: int | None = 50000,
        seed: int = 42
    ) -> pd.DataFrame:

    # Parameters validation and initializations
    model_spec = get_model_spec(model)
    train_treebank_spec = get_treebank_spec(train_treebank)
    if eval_treebanks is None:
        pud_by_lang = {"en": "pud_en", "it": "pud_it"}
        eval_treebanks = (
            train_treebank,
            pud_by_lang[train_treebank_spec["lang"]],
        )
    eval_treebank_specs = {
        treebank: get_treebank_spec(treebank)
        for treebank in eval_treebanks
    }
    if probe == "linear":
        probe_model = LinearProbe
    elif probe == "mlp":
        probe_model = MLPProbe
    else:
        raise ValueError("probe should be linear or mlp.")
    if init not in ["pretrained", "random"]:
        raise ValueError("init should be pretrained or random.")
    if pooling not in ["first", "last", "mean"]:
        raise ValueError("pooling should be first, last, or mean.")
    if not tasks:
        raise ValueError("tasks should be random or .")
    if layers is None:
        layers = list(range(model_spec["num_layers"] + 1))
    if not layers:
        raise ValueError("layers shouldn't be empty.")
    for layer in layers:
        if layer < 0 or layer > model_spec["num_layers"]:
            raise ValueError("layers value should be positive or zero.")
    layers = list(set(layers)).sort()
    if max_train_rows is not None and max_train_rows <= 0:
        raise ValueError("max_train_rows should be positive.")
    if seed < 0:
        raise ValueError("seed should be positive or zero.")
    
    control = load_control(train_treebank) if "control" in tasks else None
    
    # Build training data
    train_rows, y_train = _prepare_split(model, init, pooling, train_treebank, "train", tasks, control)
    if max_train_rows is not None and len(train_rows) > max_train_rows:
        rng = np.random.default_rng(seed)
        positions = rng.choice(
            len(train_rows),
            size=max_train_rows,
            replace=False,
        )
        train_rows = train_rows[positions]
        y_train = {
            task: labels[positions]
            for task, labels in y_train.items()
        }
    
    # Build development data
    dev_rows, y_dev = _prepare_split(model, init, pooling, dev_treebank, "dev", tasks, control)
    if max_train_rows is not None and len(train_rows) > max_train_rows:
        rng = np.random.default_rng(seed)
        positions = rng.choice(
            len(train_rows),
            size=max_train_rows,
            replace=False,
        )
        train_rows = train_rows[positions]
        y_train = {
            task: labels[positions]
            for task, labels in y_train.items()
        }
    
    # Build testing data
    test_rows, y_test = _prepare_split(model, init, pooling, test_treebank, "test", tasks, control)
    if max_train_rows is not None and len(train_rows) > max_train_rows:
        rng = np.random.default_rng(seed)
        positions = rng.choice(
            len(train_rows),
            size=max_train_rows,
            replace=False,
        )
        train_rows = train_rows[positions]
        y_train = {
            task: labels[positions]
            for task, labels in y_train.items()
        }

    return
