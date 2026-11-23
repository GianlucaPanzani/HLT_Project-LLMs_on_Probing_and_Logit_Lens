import io
import json
import os
import warnings
from collections import Counter
from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd

from lib.control_tasks import control_labels, control_next_labels
from lib.data import UPOS_TAGS
from lib.paths import probing_result_path
from lib.registry import get_model_spec, get_treebank_spec
from lib.remote_features import HfRangeSource, RemoteBatch, iter_remote_feature_batches
from ws1_probing.probes import LinearProbe, MLPProbe, Probe

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
) -> tuple[pd.DataFrame, np.ndarray, Counter[str], set[str]]:
    """Scan layer 0, retaining all words and recording feature validity.

    Parameters
    ----------
    model : str
        Model registry key used to locate the remote features.
    init : str
        Feature initialization, either 'pretrained' or 'random'.
    pooling : str
        Feature pooling mode: 'first', 'mean' or 'last'.
    treebank : str
        Treebank registry key.
    split : str
        Split to scan: 'train', 'dev' or 'test'.
    source : HfRangeSource
        Remote repository source resolved to a fixed commit.
    collect_control_enabled : bool
        If True, collect UPOS token counts and case-sensitive word types
        before filtering NaN feature rows. Defaults to False.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, Counter[str], set[str]]
        Complete streamed word table, indexed by original row positions;
        boolean feature-validity mask aligned with the table's row order;
        train UPOS counts; and distinct train word forms.
        Counts and forms are empty when collection is disabled.

    Raises
    ------
    ValueError
        If a non-null UPOS tag is unknown, control statistics are requested
        with null UPOS labels, or no rows with valid features are available.
    FileNotFoundError
        If a required remote feature, metadata or word-table file is missing.
    """

    table_parts = []
    feature_masks = []
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
        # Select the columns that, for each row/word, describe respectively:
        # - sent_id: ID of the sentence containing the word,
        # - word_idx: position of the word within the sentence (range [0,len(s))),
        # - form: original word text,
        # - upos: Universal Part-of-Speech tag of the current word,
        # - next_upos: UPOS tag of the next word in the same sentence (null for the last word).
        table = batch.table[["sent_id", "word_idx", "form", "upos", "next_upos"]].copy()
        table.index = pd.Index(batch.rows)

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

        # Keep all words without filtering them
        table_parts.append(table)

        # Record feature validity separately from label validity
        valid = ~np.isnan(batch.features).any(axis=1)
        feature_masks.append(valid)

    if not table_parts:
        raise ValueError(f"No valid rows for {treebank}/{split}.")
    
    tables_df = pd.concat(table_parts)
    valid_features = np.concatenate(feature_masks)

    if not valid_features.any():
        raise ValueError(f"No valid rows for {treebank}/{split}.")

    return tables_df, valid_features, tag_counts, word_types


def _complete_train_statistics(source: HfRangeSource, treebank: str) -> tuple[Counter[str], set[str]]:
    """Read the UPOS counts and word types of the complete train word table.

    Used when the train features are capped (``max_rows`` in ``meta.json``, mGPT): the
    control mapping must come from the complete training split, like the one of every
    other model, and the capped features only cover its first rows. The word table is
    small (about 1 MB) and is read with one range request.

    Parameters
    ----------
    source : HfRangeSource
        Remote repository source resolved to a fixed commit.
    treebank : str
        Training treebank registry key.

    Returns
    -------
    tuple[Counter[str], set[str]]
        UPOS token counts and case-sensitive word types of the whole train split.

    Raises
    ------
    ValueError
        If a UPOS tag is null or unknown.
    FileNotFoundError
        If the word table is not in the repository.
    """
    path = f"data/processed/{treebank}_train.parquet"
    table = pd.read_parquet(io.BytesIO(source.read(path, 0, source.size(path))), columns=["form", "upos"])
    if table["upos"].isna().any():
        raise ValueError("Control mapping requires non-null train UPOS.")
    if not table["upos"].isin(UPOS_TAGS).all():
        raise ValueError(f"Unknown UPOS labels in {treebank}/train.")
    return Counter(table["upos"]), set(table["form"])


def _build_control_mapping(
    train_treebank: str,
    counts: Counter[str],
    word_types: set[str],
) -> dict:
    """Build the shared control mapping from complete train statistics.

    Parameters
    ----------
    train_treebank : str
        Training treebank registry key stored in the mapping.
    counts : Counter[str]
        UPOS token counts from the complete training split, including rows
        whose features are NaN, before any training subsampling.
    word_types : set[str]
        All case-sensitive word forms in the same training split.

    Returns
    -------
    dict
        Control data containing 'treebank', 'seed', 'distribution' and
        'mapping'. Each word type receives a tag drawn from the empirical
        UPOS distribution, using seed 42 and sorted word types.

    Raises
    ------
    ValueError
        If the total token count is zero.
    """

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
        batches: Iterable[RemoteBatch],
        layer: int,
        rows: np.ndarray | pd.Index | list[int],
        treebank: str,
        hidden_size: int,
        split: str,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield selected feature batches with positions aligned to requested rows.

    Parameters
    ----------
    batches : Iterable[RemoteBatch]
        Feature batches for one layer and split, with original word-table
        row indices. Each source row is expected to occur only once.
    layer : int
        Layer index, used in validation error messages.
    rows : np.ndarray | pd.Index | list[int]
        Unique original word-table row indices to select. Their order defines
        the positions returned alongside each feature batch.
    treebank : str
        Treebank registry key, used in validation error messages.
    hidden_size : int
        Expected feature dimension from the model registry.
    split : str
        Split name, used in validation error messages.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        Integer positions into 'rows' and the corresponding float32 feature
        matrix of shape (n_selected_in_batch, hidden_size). Batches without
        selected rows are skipped; batches remain in source iteration order.

    Raises
    ------
    ValueError
        If selected features have the wrong dimension or nonfinite values,
        or any requested row is missing when the iterator is exhausted.
    """

    rows_index = pd.Index(rows)  # Rows indexes
    seen = np.zeros(len(rows_index), dtype=bool)  # Zeros array

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

        seen[selected_positions] = True
        yield selected_positions, features
    
    if not seen.all():
        raise ValueError(f"Missing selected feature rows for {treebank}/{split}, layer {layer}.")
    return

def _evaluate_probes(
        model: str,
        init: str,
        pooling: str,
        eval_treebank: str,
        layer: int,
        rows_index: pd.Index,
        y_eval_dict: dict[str, np.ndarray],
        fitted_probes: dict[str, Probe],
        hidden_size: int,
        *,
        hf_source: HfRangeSource,
    ) -> tuple[dict[str, int], int]:
    """Evaluate fitted probes on one test treebank and layer in feature batches.

    Parameters
    ----------
    model : str
        Model registry key used to locate the features.
    init : str
        Feature initialization, either 'pretrained' or 'random'.
    pooling : str
        Feature pooling mode: 'first', 'mean' or 'last'.
    eval_treebank : str
        Registry key of the treebank whose test split is evaluated.
    layer : int
        Layer index to read, including 0 for embeddings.
    rows_index : pd.Index
        Unique original word-table indices of valid evaluation rows.
    y_eval_dict : dict[str, np.ndarray]
        Integer class labels for each task, with one label per entry of
        rows_index in the same order. Must include every fitted probe's task.
    fitted_probes : dict[str, Probe]
        Trained probes keyed by task name. Each probe predicts UPOS indices.
    hidden_size : int
        Expected feature dimension from the model registry.
    hf_source : HfRangeSource
        Remote repository source resolved to the experiment's fixed commit.

    Returns
    -------
    tuple[dict[str, int], int]
        Number of correct predictions per task and total evaluated rows.
        Feature batches are converted to float32 and discarded after use.

    Raises
    ------
    ValueError
        If selected features are invalid, requested rows are missing, or
        the evaluated row count differs from len(rows_index).
    FileNotFoundError
        If a required remote feature, metadata or word-table file is missing.
    """
    batches = iter_remote_feature_batches(
        hf_source.repo_id,
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
        hidden_size=hidden_size,
        split="test",
    )
    
    # Predictions
    correct_pred_counter = {task: 0 for task in fitted_probes.keys()}
    total_predictions = 0
    for positions, features in selected_features_iterator:
        for task, probe_model in fitted_probes.items():
            predictions = probe_model.predict(features)
            labels = y_eval_dict[task][positions]
            correct_pred_counter[task] += int(np.count_nonzero(predictions == labels))
        total_predictions += len(features)

    if total_predictions != len(rows_index):
        raise ValueError(f"Missing evaluation rows for {eval_treebank}, layer {layer}.")
    
    return correct_pred_counter, total_predictions


def _get_task_labels(
    table: pd.DataFrame,
    task: str,
    control: dict | None,
) -> pd.Series:
    """Get task labels before filtering or sampling word rows.

    Parameters
    ----------
    table : pd.DataFrame
        Complete word table in original sentence and word order.
    task : str
        One of 'upos', 'control', 'upos_next' or 'control_next'.
    control : dict or None
        Training control mapping; required for control tasks.

    Returns
    -------
    pd.Series
        Labels aligned with the table index, retaining null targets.
    """
    if task == "upos":
        return table["upos"]
    if task == "upos_next":
        return table["next_upos"]

    if task in ("control", "control_next"):
        if control is None:
            raise ValueError(f"Task '{task}' requires a control mapping.")
        if task == "control":
            return control_labels(table, control)
        return control_next_labels(table, control)

    raise ValueError(f"Unknown task: '{task}'.")


def _select_valid_data(
    labels_by_task: dict[str, pd.Series],
    valid_features: np.ndarray,
    *,
    max_rows: int | None = None,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Select valid word rows and encode labels, sharing rows within task pairs.

    Parameters
    ----------
    labels_by_task : dict[str, pd.Series]
        Task labels computed on the complete table, with original row indices.
    valid_features : np.ndarray
        Boolean feature-validity mask aligned with the label series.
    max_rows : int or None
        Maximum selected rows per task pair; None keeps all valid rows.
    seed : int
        Seed for reproducible sampling without replacement.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        Each task maps to original word-table row indices and integer labels,
        both in the same order.
    """
    task_data = {}
    tag_to_id = {tag: i for i, tag in enumerate(UPOS_TAGS)}

    for task_pair in (("upos", "control"), ("upos_next", "control_next")):
        active_tasks = [task for task in task_pair if task in labels_by_task]
        if not active_tasks:
            continue

        # Use the same valid rows for the task and its control
        valid = valid_features.copy()
        for task in active_tasks:
            labels = labels_by_task[task]
            unknown = labels.notna() & ~labels.isin(UPOS_TAGS)
            if unknown.any():
                raise ValueError(f"Unknown UPOS labels for task '{task}'.")
            valid &= labels.notna().to_numpy()

        valid_positions = np.flatnonzero(valid)
        if len(valid_positions) == 0:
            raise ValueError(f"No valid rows for tasks {active_tasks}.")

        # Sampling on valid rows once per pair
        if max_rows is not None and len(valid_positions) > max_rows:
            rng = np.random.default_rng(seed)
            valid_positions = rng.choice(valid_positions, size=max_rows, replace=False)

        for task in active_tasks:
            selected_labels = labels_by_task[task].iloc[valid_positions]
            selected_rows = selected_labels.index.to_numpy(dtype=np.int64)

            labels_ids = selected_labels.map(tag_to_id).to_numpy(dtype=np.int64)
            task_data[task] = (selected_rows, labels_ids)

    return task_data

# -----------------------------------------------------------------------------
# Public entry point: train and evaluate probes, then save the results.
# -----------------------------------------------------------------------------

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
    model : str
        Model registry key; layer count, hidden size and kind come from its spec.
    train_treebank : str
        Training treebank registry key, with train, dev and test splits.
    probe : str
        Probe architecture: 'linear' or 'mlp'. A new probe is fitted for
        each layer and task, with hyperparameters selected on dev.
    init : str
        Feature initialization: 'pretrained' or 'random'.
    pooling : str
        Feature pooling mode: 'first', 'mean' or 'last'.
    tasks : tuple[str, ...]
        Nonempty sequence of unique tasks: 'upos', 'control', 'upos_next'
        or 'control_next'. Defaults to ('upos', 'control').
    eval_treebanks : tuple[str, ...] | None
        Nonempty sequence of unique test treebanks. None selects the training
        treebank and its language's PUD; missing evaluation artifacts are skipped.
    layers : list[int] | None
        Unique layer indices in 0..L. None selects all layers, including embeddings.
    max_train_rows : int | None
        Positive maximum number of valid training rows per task pair
        (default: 50000), or None for all valid rows. Sampling is shared
        within each task/control pair and across layers.
        Dev and evaluation rows are not subsampled.
    seed : int
        Non-negative sampling and probe seed (default: 42). The shared control
        mapping always uses seed 42, independently of this argument.
    verbose : int
        If nonzero, print per-layer evaluation accuracies and the output path.
        Defaults to 0, which disables these progress messages.

    Returns
    -------
    pd.DataFrame
        Results following CONTRACTS.md section 6, with one row per evaluation
        treebank, task and layer for the given seed. The same frame overwrites
        probing_result_path(model, train_treebank, probe, init, pooling).

    Raises
    ------
    ValueError
        If arguments or feature data are invalid, HLT_HF_REVISION is unset,
        training features are capped when rebuilding control labels, or no
        evaluation treebank remains available.
    FileNotFoundError
        If required training or development artifacts, or the remote revision,
        are missing.
    PermissionError
        If access to the remote repository or its artifacts is denied.

    Warns
    -----
    RuntimeWarning
        When an evaluation treebank is skipped because an artifact is missing.
        Any results already collected for that treebank are discarded.

    Notes
    -----
    HLT_HF_REPO selects the remote repository (default: 'hlt-group11/features').
    HLT_HF_REVISION is required and resolved to a fixed commit for the run.
    Layer 0 defines the NaN mask shared by all layers; null UPOS rows are dropped.
    Train and dev features are held for one layer at a time as float32 arrays;
    evaluation features are streamed in batches.
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
    allowed_tasks = ("upos", "control", "upos_next", "control_next")
    if not tasks or any(task not in allowed_tasks for task in tasks):
        raise ValueError(f"tasks must contain only values from {allowed_tasks}.")
    if len(set(tasks)) != len(tasks):
        raise ValueError("tasks must not contain duplicates.")
    if (
        model_spec["kind"] == "decoder" 
        and ("upos_next" in tasks or "control_next" in tasks)  # has a next task
        and pooling != "last"
    ):
        raise ValueError("Next-word tasks require pooling='last' for decoders.")
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
    
    # Get HuggingFace repo information from environment variables
    repo_id = os.environ.get("HLT_HF_REPO", "hlt-group11/features")
    revision = os.environ.get("HLT_HF_REVISION")
    if not revision:
        raise ValueError("Set HLT_HF_REVISION to the feature repository commit.")
    hf_source = HfRangeSource(repo_id, revision)

    # Capped features still need control statistics from the complete train split.
    control_is_needed = "control" in tasks or "control_next" in tasks
    capped_train = False
    if control_is_needed:
        meta_path = f"{model}/{init}/{pooling}/{train_treebank}_train/meta.json"
        meta = json.loads(hf_source.read(meta_path, 0, hf_source.size(meta_path)))
        capped_train = meta.get("max_rows") is not None

    if verbose:
        print("Building datasets from HuggingFace repo...")
    
    # Read all training words (keep feature validity separate)
    train_table, train_valid_features, upos_tag_counts, word_types = _scan_data(
        model,
        init,
        pooling,
        train_treebank,
        "train",
        source=hf_source,
        collect_control_enabled=control_is_needed and not capped_train,
    )

    if capped_train:
        upos_tag_counts, word_types = _complete_train_statistics(hf_source, train_treebank)

    # Build the control mapping from the complete training statistics.
    control = None
    if control_is_needed:
        control = _build_control_mapping(
            train_treebank,
            upos_tag_counts,
            word_types,
        )
    
    # Build development data
    dev_table, dev_valid_features, _, _ = _scan_data(
        model,
        init,
        pooling,
        train_treebank,
        "dev",
        source=hf_source
    )

    # Build testing data
    eval_tables_dict = {}
    eval_valid_features_dict = {}
    for eval_treebank in eval_treebanks:
        try:
            # Check missing layers before fitting any probes
            for layer in layers:
                path = f"{model}/{init}/{pooling}/{eval_treebank}_test/layer_{layer:02d}.npy"
                hf_source.size(path)
                
            # Build testing table
            eval_table, eval_valid_features, _, _ = _scan_data(
                model,
                init,
                pooling,
                eval_treebank,
                "test",
                source=hf_source
            )
        except FileNotFoundError as error:
            warnings.warn(f"Skipping {eval_treebank}: {error}", RuntimeWarning, stacklevel=2)
            continue

        eval_tables_dict[eval_treebank] = eval_table
        eval_valid_features_dict[eval_treebank] = eval_valid_features

    if not eval_tables_dict:
        raise ValueError("No evaluation treebank with available features remains.")

    # Compute labels for all the tables
    train_labels_dict = {}
    dev_labels_dict = {}
    eval_labels_dict = {treebank: {} for treebank in eval_tables_dict}
    for task in tasks:
        train_labels_dict[task] = _get_task_labels(train_table, task, control)
        dev_labels_dict[task] = _get_task_labels(dev_table, task, control)
        for eval_treebank, eval_table in eval_tables_dict.items():
            eval_labels_dict[eval_treebank][task] = _get_task_labels(eval_table, task, control)

    # Select valid rows for all the datasets
    train_data = _select_valid_data(
        train_labels_dict,
        train_valid_features,
        max_rows=max_train_rows,
        seed=seed,
    )
    dev_data = _select_valid_data(
        dev_labels_dict,
        dev_valid_features,
    )
    eval_data = {}
    for eval_treebank in eval_tables_dict:
        eval_data[eval_treebank] = _select_valid_data(
            eval_labels_dict[eval_treebank],
            eval_valid_features_dict[eval_treebank],
        )

    if verbose:
        print("Training phase:")

    # Iteration over the hidden layers
    result_rows = []
    for layer in layers:
        if verbose:
            print(f"  Layer {layer}", end=" ")

        # Train each task using its selected word rows and aligned targets
        fitted_probes = {}
        for task in tasks:
            X_train_rows, y_train = train_data[task]
            X_dev_rows, y_dev = dev_data[task]

            # Load the selected train and dev features for this layer and task.
            features_split = {}
            for split, rows in (("train", X_train_rows), ("dev", X_dev_rows)):
                hidden_size = model_spec["hidden_size"]
                selected_features = np.empty((len(rows), hidden_size), dtype=np.float32)

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
                    hidden_size=hidden_size,
                    split=split,
                )
                for positions, features in selected_features_iterator:
                    selected_features[positions] = features
                
                # Store the features which will be used for that split
                features_split[split] = selected_features
                del selected_features

            # Training of the probe model with the split created
            fitted_probes[task] = probe_model(seed=seed).fit(
                features_split["train"],
                y_train,
                features_split["dev"],
                y_dev,
            )
            del features_split

        if verbose:
            print(" trained")
            print("  Evaluation phase:")

        # Evaluate each task on its selected test rows.
        for eval_treebank, task_data in list(eval_data.items()):
            task_scores = {}

            try:
                for task in tasks:
                    X_eval_rows, y_eval = task_data[task]

                    correct_pred_counter, total_predictions = _evaluate_probes(
                        model,
                        init,
                        pooling,
                        eval_treebank,
                        layer,
                        rows_index=pd.Index(X_eval_rows),
                        y_eval_dict={task: y_eval},
                        fitted_probes={task: fitted_probes[task]},
                        hidden_size=model_spec["hidden_size"],
                        hf_source=hf_source,
                    )

                    task_scores[task] = (correct_pred_counter[task], total_predictions)

            except FileNotFoundError as error:
                warnings.warn(f"Skipping {eval_treebank}: {error}", RuntimeWarning, stacklevel=2)
                del eval_data[eval_treebank]
                result_rows = [row for row in result_rows if row["eval_treebank"] != eval_treebank]
                continue
            
            # Outputs
            if verbose:
                print(f"    Layer {layer}, {eval_treebank}:", end=" ")
            for task in tasks:
                correct_predictions, total_predictions = task_scores[task]
                accuracy = correct_predictions / total_predictions

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
                    "acc": accuracy,
                    "n_train": len(train_data[task][0]),
                    "n_eval": total_predictions,
                    "hparams": json.dumps(fitted_probes[task].hparams, sort_keys=True),
                    "seed": seed,
                })
                if verbose:
                    print(f"{task}={accuracy:.4f}", end=" ")
            if verbose:
                print()

        del fitted_probes

        if not eval_data:
            raise ValueError("No evaluation treebank with available features remains.")

    # Create results path
    output_path = probing_result_path(model, train_treebank, probe, init, pooling)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save data
    results = pd.DataFrame(result_rows, columns=list(_result_types)).astype(_result_types)
    results.to_parquet(output_path, index=False)

    return results
