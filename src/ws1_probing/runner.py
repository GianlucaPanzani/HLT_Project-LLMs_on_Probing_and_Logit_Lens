from collections import Counter
import numpy as np
import os
import pandas as pd

from lib.control_tasks import control_labels
from lib.data import UPOS_TAGS
from lib.registry import get_model_spec, get_treebank_spec
from lib.remote_features import HfRangeSource, iter_remote_feature_batches
from ws1_probing.probes import LinearProbe, MLPProbe


def _scan_data(
    model: str,
    init: str,
    pooling: str,
    treebank: str,
    split: str,
    *,
    source: HfRangeSource,
    collect_control_enabled: bool = False
) -> tuple[pd.DataFrame, Counter[str], set[str]]:

    table_parts = []
    tag_counts = Counter()
    word_types = set()

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
        table = batch.table[["form", "upos"]]

        # Check if there are unknown labels
        unknown = table["upos"].notna() & ~table["upos"].isin(UPOS_TAGS)
        if unknown.any():
            raise ValueError(f"Unknown UPOS labels in {treebank}/{split}.")

        # Save UPOS tag and form
        if collect_control_enabled:
            if table["upos"].isna().any():
                raise ValueError("Control mapping requires non-null train UPOS.")
            tag_counts.update(table["upos"])
            word_types.update(table["form"])

        # Save only valid data (non-NaN data) from the whole table
        valid = ~np.isnan(batch.features).any(axis=1) & table["upos"].notna().to_numpy()
        if valid.any():
            selected = table.iloc[np.flatnonzero(valid)].copy()
            selected.index = batch.rows[valid]
            table_parts.append(selected)

    # Build the Dataframe
    if not table_parts:
        raise ValueError(f"No valid rows for {treebank}/{split}.")
    tables_df = pd.concat(table_parts)

    return tables_df, tag_counts, word_types

def _build_control_mapping(
    train_treebank: str,
    counts: Counter[str],
    word_types: set[str],
) -> dict:
    """Build the shared control mapping from complete train statistics."""

    total = sum(counts.values())
    if total == 0:
        raise ValueError("Cannot build control mapping from empty statistics.")

    # Build the frequency of each UPOS tag
    distribution = {
        tag: counts.get(tag, 0) / total
        for tag in UPOS_TAGS
    }

    # Build the probabilities from the frequecies
    probabilities = np.array(
        [distribution[tag] for tag in UPOS_TAGS],
        dtype=np.float64,
    )

    # Tags are drawn randomly from the probability distribution
    forms = sorted(word_types)
    rng = np.random.default_rng(42)
    drawn_labels = rng.choice(
        len(UPOS_TAGS),
        size=len(forms),
        p=probabilities / probabilities.sum(),
    )

    return {
        "treebank": train_treebank,
        "seed": 42,
        "distribution": distribution,
        "mapping": {
            form: UPOS_TAGS[int(label)]
            for form, label in zip(forms, drawn_labels)
        },
    }


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

    # Initializations
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
    
    # Parameters validation
    if init not in ["pretrained", "random"]:
        raise ValueError("init should be pretrained or random.")
    if pooling not in ["first", "last", "mean"]:
        raise ValueError("pooling should be first, last, or mean.")
    if not tasks or any(task not in ("upos", "control") for task in tasks):
        raise ValueError("tasks must contain only 'upos' and 'control'.")
    if len(set(tasks)) != len(tasks):
        raise ValueError("tasks must not contain duplicates.")
    if layers is None:
        layers = list(range(model_spec["num_layers"] + 1))
    if not layers:
        raise ValueError("layers shouldn't be empty.")
    for layer in layers:
        if layer < 0 or layer > model_spec["num_layers"]:
            raise ValueError("layers value should be positive or zero.")
    if len(set(layers)) != len(layers):
        raise ValueError("layers must not contain duplicates.")
    if max_train_rows is None or max_train_rows <= 0:
        raise ValueError("max_train_rows should be positive.")
    if seed < 0:
        raise ValueError("seed should be positive or zero.")
    
    # Get the HuggingFace repository info from environment variables
    repo_id = os.environ.get("HLT_HF_REPO", "hlt-group11/features")
    revision = os.environ.get("HLT_HF_REVISION")
    if not revision:
        raise ValueError("Set HLT_HF_REVISION to the feature repository commit.")
    hf_source = HfRangeSource(repo_id, revision)
    
    # Build training data
    train_table, upos_tag_counts, word_types = _scan_data(
        model,
        init,
        pooling,
        train_treebank,
        "train",
        source=hf_source,
        collect_control_enabled="control" in tasks
    )
    if len(train_table) > max_train_rows:
        rng = np.random.default_rng(seed)
        drawn_positions = rng.choice(
            len(train_table),
            size=max_train_rows,
            replace=False,
        )
        train_table = train_table.iloc[drawn_positions].copy()
    train_rows = train_table.index.to_numpy(dtype=np.int64)

    # Build the control tags map
    control = None
    if "control" in tasks:
        control = _build_control_mapping(train_treebank, upos_tag_counts, word_types)
    
    # Build development data
    dev_table, _, _ = _scan_data(
        model,
        init,
        pooling,
        train_treebank,
        "dev",
        source=hf_source
    )
    dev_rows = dev_table.index.to_numpy(dtype=np.int64)

    # Build the integer labels for each dataset (train and dev)
    tag_to_id_map = {tag: i for i, tag in enumerate(UPOS_TAGS)}
    y_train = {}
    y_dev = {}
    for task in tasks:
        if task == "upos":
            train_labels = train_table["upos"]
            dev_labels = dev_table["upos"]
        if task == "control":
            train_labels = control_labels(train_table, control)
            dev_labels = control_labels(dev_table, control)
        y_train[task] = train_labels.map(tag_to_id_map).to_numpy(dtype=np.int64)
        y_dev[task] = dev_labels.map(tag_to_id_map).to_numpy(dtype=np.int64)

    return
