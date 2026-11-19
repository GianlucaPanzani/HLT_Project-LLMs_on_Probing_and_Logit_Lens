import json
import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from lib.control_tasks import control_labels
from lib.data import UPOS_TAGS
from lib.paths import probing_result_path
from lib.registry import get_model_spec, get_treebank_spec
from lib.remote_features import HfRangeSource, iter_remote_feature_batches
from ws1_probing.probes import LinearProbe, MLPProbe


_result_types = {
    "model": "string",
    "train_treebank": "string",
    "eval_treebank": "string",
    "lang": "string",
    "kind": "string",
    "probe": "string",
    "task": "string",
    "init": "string",
    "pooling": "string",
    "layer": "int64",
    "L": "int64",
    "tau": "float64",
    "acc": "float64",
    "n_train": "int64",
    "n_eval": "int64",
    "hparams": "string",
    "seed": "int64",
}


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


def _select_features_iterator(
        batches,
        layer,
        rows,
        treebank,
        hidden_size,
        split,
    ):

    rows_index = pd.Index(rows)  # Rows indexes

    for batch in batches:

        # Select the positive positions only
        positions = rows_index.get_indexer(batch.rows)
        keep_mask = positions >= 0
        if not keep_mask.any():
            continue
        
        # Select positions and relative features
        selected_positions = positions[keep_mask]
        features = np.asarray(batch.features[keep_mask], dtype=np.float32)
        if features.shape[1] != hidden_size or not np.isfinite(features).all():
            raise ValueError(f"Invalid selected features for {treebank}/{split}, layer {layer}.")

        yield selected_positions, features
        


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
        seed: int = 42,
        verbose: int = 0,
    ) -> pd.DataFrame:
    """Train probes layer by layer using remote features and save test results.

    Parameters
    ----------
    model, train_treebank : str
        Registry keys. The training treebank must have train, dev and test splits.
    probe, init, pooling : str
        Probe architecture and feature configuration.
    tasks : tuple[str, ...]
        UPOS and/or control tasks.
    eval_treebanks : tuple[str, ...] | None
        Test treebanks; defaults to the training treebank and its language's PUD.
    layers : list[int] | None
        Layers to probe; defaults to all layers including embeddings.
    max_train_rows : int | None
        Maximum number of valid training rows, or None for all valid rows.
    seed : int
        Sampling and probe seed. The shared control mapping uses seed 42.

    Returns
    -------
    pd.DataFrame
        Results following CONTRACTS.md section 6, also written to parquet.
        HLT_HF_REVISION selects the remote commit; HLT_HF_REPO selects the repo.
    """

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
    if not {"train", "dev", "test"}.issubset(train_treebank_spec["splits"]):
        raise ValueError("train_treebank must have train, dev and test splits.")
    if not eval_treebanks or len(set(eval_treebanks)) != len(eval_treebanks):
        raise ValueError("eval_treebanks must be nonempty and contain no duplicates.")
    if any("test" not in spec["splits"] for spec in eval_treebank_specs.values()):
        raise ValueError("Every evaluation treebank must have a test split.")
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
        if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer <= model_spec["num_layers"]:
            raise ValueError(f"layers must contain integers in [0, {model_spec['num_layers']}].")
    if len(set(layers)) != len(layers):
        raise ValueError("layers must not contain duplicates.")
    if max_train_rows is not None:
        if isinstance(max_train_rows, bool) or not isinstance(max_train_rows, int) or max_train_rows <= 0:
            raise ValueError("max_train_rows must be a positive integer or None.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    
    # Get the HuggingFace repository info from environment variables
    repo_id = os.environ.get("HLT_HF_REPO", "hlt-group11/features")
    revision = os.environ.get("HLT_HF_REVISION")
    if not revision:
        raise ValueError("Set HLT_HF_REVISION to the feature repository commit.")
    hf_source = HfRangeSource(repo_id, revision)

    # A control mapping cannot be reconstructed from a possibly capped train split
    if "control" in tasks:
        meta_path = f"{model}/{init}/{pooling}/{train_treebank}_train/meta.json"
        meta = json.loads(hf_source.read(meta_path, 0, hf_source.size(meta_path)))
        if meta.get("max_rows") is not None:
            raise ValueError(
                "Cannot rebuild control labels from capped training features. "
                "Use a mapping built from the complete training word table."
            )
    
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
    if max_train_rows is not None and len(train_table) > max_train_rows:
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

    # Keep only row indices and labels for evaluation: feature batches are discarded.
    eval_tables = {}
    for eval_treebank in eval_treebanks:

        # Check missing layers before fitting any probes
        try:
            for layer in layers:
                hf_source.size(f"{model}/{init}/{pooling}/{eval_treebank}_test/layer_{layer:02d}.npy")
        except FileNotFoundError as error:
            warnings.warn(f"Skipping {eval_treebank}: {error}", RuntimeWarning, stacklevel=2)
            continue
        
        # Build testing data
        eval_table, _, _ = _scan_data(
            model,
            init,
            pooling,
            eval_treebank,
            "test",
            source=hf_source
        )
        eval_tables[eval_treebank] = eval_table

    # Build the integer labels for each dataset (train, dev and test)
    tag_to_id_map = {tag: i for i, tag in enumerate(UPOS_TAGS)}
    y_train = {}
    y_dev = {}
    y_eval_data = {treebank: (pd.Index(table.index), {}) for treebank, table in eval_tables.items()}
    for task in tasks:
        if task == "upos":
            train_labels = train_table["upos"]
            dev_labels = dev_table["upos"]
        if task == "control":
            train_labels = control_labels(train_table, control)
            dev_labels = control_labels(dev_table, control)
        y_train[task] = train_labels.map(tag_to_id_map).to_numpy(dtype=np.int64)
        y_dev[task] = dev_labels.map(tag_to_id_map).to_numpy(dtype=np.int64)
        for eval_treebank, eval_table in eval_tables.items():
            if task == "upos":
                eval_labels = eval_table["upos"]
            if task == "control":
                eval_labels = control_labels(eval_table, control)
            y_eval_data[eval_treebank][1][task] = eval_labels.map(tag_to_id_map).to_numpy(dtype=np.int64)
            del eval_table

    if not y_eval_data:
        raise ValueError("No evaluation treebank with available features remains.")

    del train_table, dev_table, eval_tables, upos_tag_counts, word_types

    # Iteration over the hidden layers
    result_rows = []
    for layer in layers:

        # Pick data from HuggingFace repo
        split_features = {}
        for split, rows in (("train", train_rows), ("dev", dev_rows)):
            n_rows = len(rows)

            batches = iter_remote_feature_batches(
                repo_id,
                model,
                init,
                pooling,
                train_treebank,
                split,
                layer,
                revision=hf_source.revision,
                source=hf_source,
            )

            selected_features_iterator = _select_features_iterator(
                batches,
                layer,
                rows=rows,
                treebank=train_treebank,
                hidden_size=model_spec["hidden_size"],
                split=split,
            )

            selected_features = np.empty((n_rows, model_spec["hidden_size"]), dtype=np.float32)  # Empty NumPy array
            for positions, features in selected_features_iterator:
                selected_features[positions] = features
            split_features[split] = selected_features

            del selected_features

        # Training probes (for both "upos" and "control" if needed)
        fitted_probes = {}
        for task in tasks:
            fitted_probes[task] = probe_model(seed=seed).fit(
                split_features["train"],
                y_train[task],
                split_features["dev"],
                y_dev[task]
            )
        del split_features

        # Evaluating probes
        for eval_treebank, (rows_index, y_eval_dict) in list(y_eval_data.items()):

            batches = iter_remote_feature_batches(
                repo_id,
                model,
                init,
                pooling,
                eval_treebank,
                "test",
                layer,
                revision=hf_source.revision,
                source=hf_source,
            )

            selected_features_iterator = _select_features_iterator(
                batches,
                layer,
                rows=rows_index,
                treebank=eval_treebank,
                hidden_size=model_spec["hidden_size"],
                split="test",
            )
            
            # Predictions
            correct_pred_counter = {task: 0 for task in tasks}
            total_predictions = 0
            for positions, features in selected_features_iterator:
                for task in tasks:
                    predictions = fitted_probes[task].predict(features)
                    labels = y_eval_dict[task][positions]
                    correct_pred_counter[task] += int(np.count_nonzero(predictions == labels))
                total_predictions += len(features)

            if total_predictions != len(rows_index):
                raise ValueError(f"Missing evaluation rows for {eval_treebank}, layer {layer}.")
            
            # Outputs
            if verbose:
                print(f"Layer {layer}, {eval_treebank}:", end=" ")
            for task in tasks:
                result_rows.append({
                    "model": model,
                    "train_treebank": train_treebank,
                    "eval_treebank": eval_treebank,
                    "lang": eval_treebank_specs[eval_treebank]["lang"],
                    "kind": model_spec["kind"],
                    "probe": probe,
                    "task": task,
                    "init": init,
                    "pooling": pooling,
                    "layer": layer,
                    "L": model_spec["num_layers"],
                    "tau": layer / model_spec["num_layers"],
                    "acc": correct_pred_counter[task] / total_predictions,
                    "n_train": len(train_rows),
                    "n_eval": total_predictions,
                    "hparams": json.dumps(fitted_probes[task].hparams, sort_keys=True),
                    "seed": seed,
                })
                if verbose:
                    print(f"{task}={correct_pred_counter[task] / total_predictions:.4f}", end=" ")

        del fitted_probes

        if not y_eval_data:
            raise ValueError("No evaluation treebank with available features remains.")

    # Create results path
    output_path = probing_result_path(model, train_treebank, probe, init, pooling)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save data
    results = pd.DataFrame(result_rows, columns=list(_result_types)).astype(_result_types)
    results.to_parquet(output_path, index=False)

    if verbose:
        print(f" --- Output path: {output_path} --- ")
    return results
