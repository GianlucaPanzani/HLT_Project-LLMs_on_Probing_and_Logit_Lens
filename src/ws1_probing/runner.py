import os
import numpy as np
import pandas as pd

from lib.control_tasks import control_labels, load_control
from lib.data import UPOS_TAGS
from lib.registry import get_model_spec, get_treebank_spec
from lib.remote_features import HfRangeSource, iter_remote_feature_batches
from ws1_probing.probes import LinearProbe, MLPProbe


def _prepare_split(
    model: str,
    init: str,
    pooling: str,
    treebank: str,
    split: str,
    tasks: tuple[str, ...],
    control: dict | None,
    *,
    source: HfRangeSource
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return valid row positions and aligned integer labels."""
    table_parts = []
    row_parts = []
    valid_parts = []

    # Pick the batches from HF
    batches = iter_remote_feature_batches(
        repo_id=source.repo_id,
        model=model,
        init=init,
        pooling=pooling,
        treebank=treebank,
        split=split,
        layer=0,
        revision=source.revision,
        source=source,
    )
    for batch in batches:
        table_parts.append(batch.table)
        row_parts.append(batch.rows)
        valid_parts.append(~np.isnan(batch.features).any(axis=1))

    if not row_parts:
        raise ValueError(f"No feature rows for {treebank}/{split}.")

    table = pd.concat(table_parts)
    original_rows = np.concatenate(row_parts)
    valid = np.concatenate(valid_parts)

    tag_to_id = {tag: index for index, tag in enumerate(UPOS_TAGS)}
    encoded_labels = {}
    for task in tasks:
        # Get the labels based on the type of task
        if task == "upos":
            labels = table["upos"]
        elif task == "control":
            if control is None:
                raise ValueError("Control mapping is required.")
            labels = control_labels(table, control)
        else:
            raise ValueError(f"Unsupported task: {task!r}.")

        # Check of unknown labels
        unknown = labels.notna() & ~labels.isin(UPOS_TAGS)
        if unknown.any():
            raise ValueError(f"Unknown labels for {task} in {treebank}/{split}.")

        valid &= labels.notna().to_numpy()
        encoded_labels[task] = labels.map(tag_to_id)

    positions = np.flatnonzero(valid)
    if positions.size == 0:
        raise ValueError(f"No valid rows for {treebank}/{split}.")

    targets = {
        task: labels.iloc[positions].to_numpy(dtype=np.int64)
        for task, labels in encoded_labels.items()
    }
    return original_rows[positions], targets


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
    
    # Get the HuggingFace repository info from environment variables
    repo_id = os.environ.get("HLT_HF_REPO", "hlt-group11/features")
    revision = os.environ.get("HLT_HF_REVISION")
    if not revision:
        raise ValueError("Set HLT_HF_REVISION to the feature repository commit.")
    hf_source = HfRangeSource(repo_id, revision)
    
    control = load_control(train_treebank) if "control" in tasks else None
    
    # Build training data
    train_rows, y_train = _prepare_split(
        model,
        init,
        pooling,
        train_treebank,
        "train",
        tasks,
        control,
        source=hf_source
    )
    if max_train_rows is not None and len(train_rows) > max_train_rows:
        rng = np.random.default_rng(seed)
        positions = rng.choice(
            len(train_rows),
            size=max_train_rows,
            replace=False,
        )
        train_rows = train_rows[positions]
        y_train = {task: labels[positions] for task, labels in y_train.items()}
    
    # Build development data
    dev_rows, y_dev = _prepare_split(
        model,
        init,
        pooling,
        train_treebank,
        "train",
        tasks,
        control,
        source=hf_source
    )
    if max_train_rows is not None and len(train_rows) > max_train_rows:
        rng = np.random.default_rng(seed)
        positions = rng.choice(
            len(train_rows),
            size=max_train_rows,
            replace=False,
        )
        train_rows = train_rows[positions]
        y_train = {task: labels[positions] for task, labels in y_train.items()}
    
    # Build testing data
    test_rows, y_test = _prepare_split(
        model,
        init,
        pooling,
        train_treebank,
        "train",
        tasks,
        control,
        source=hf_source
    )
    if max_train_rows is not None and len(train_rows) > max_train_rows:
        rng = np.random.default_rng(seed)
        positions = rng.choice(
            len(train_rows),
            size=max_train_rows,
            replace=False,
        )
        train_rows = train_rows[positions]
        y_train = {task: labels[positions] for task, labels in y_train.items()}

    return
