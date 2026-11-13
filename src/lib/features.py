"""Feature extraction engine (CONTRACTS.md sections 4 and 10).

Runs a frozen model over the sentences of a word table and stores one vector
per UD word for every layer 0..L: one float16 .npy file per layer, rows in
word-table order, plus a meta.json written last (a folder without meta.json
is an incomplete run).
"""

import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from lib.alignment import POOLINGS, encode_words_detailed, pool_words
from lib.data import load_model
from lib.paths import feature_dir, layer_path
from lib.registry import get_model_spec
from lib.word_table import load_word_table, table_to_sentences

logger = logging.getLogger(__name__)

SEED = 42
INITS: tuple[str, ...] = ("pretrained", "random")
SAVED_DTYPE = np.float16


def default_device() -> str:
    '''
    Pick the device the same way lib.data.load_model does.

    Returns
    -------
    str: "cuda" if available, otherwise "mps", otherwise "cpu".
    '''
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_max_length(config) -> int:
    '''
    Maximum sequence length of a model, from its configuration.

    Parameters
    ----------
    - config (PretrainedConfig): Model configuration.

    Returns
    -------
    int: config.max_position_embeddings (BERT-family) or config.n_positions
    (GPT-2-family). tokenizer.model_max_length is never used because some
    tokenizers report a huge sentinel value.
    '''
    for name in ("max_position_embeddings", "n_positions"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Cannot find max_position_embeddings or n_positions in the model configuration.")


def plan_sentences(
    sentences: list[tuple[str, list[str]]],
    max_rows: int | None = None,
) -> tuple[list[tuple[str, list[str], int]], int]:
    '''
    Give each sentence its row offset in word-table order.

    Parameters
    ----------
    - sentences (list[tuple[str, list[str]]]): (sent_id, words) in row order,
    as returned by lib.word_table.table_to_sentences.
    - max_rows (int | None): If set, keep whole sentences only, as long as
    the total number of rows stays within max_rows. Defaults to None.

    Returns
    -------
    A tuple (planned, n_rows): (sent_id, words, offset) per kept sentence and
    the total number of rows covered.
    '''
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive.")
    planned = []
    offset = 0
    for sent_id, words in sentences:
        if not words:
            raise ValueError(f"Sentence {sent_id} has no words.")
        if max_rows is not None and offset + len(words) > max_rows:
            break
        planned.append((sent_id, words, offset))
        offset += len(words)
    return planned, offset


def make_batches(lengths: list[int], batch_size: int) -> list[list[int]]:
    '''
    Group sentence indices into batches of similar length.

    Parameters
    ----------
    - lengths (list[int]): Number of words of each sentence.
    - batch_size (int): Maximum number of sentences per batch.

    Returns
    -------
    list[list[int]]: Sentence indices, sorted by length (longest first, ties
    in original order) and cut into batches, so that padding is small and an
    out-of-memory error, if any, happens at the very start.
    '''
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    return [order[i: i + batch_size] for i in range(0, len(order), batch_size)]


def extract_into(
    model,
    tokenizer,
    planned: list[tuple[str, list[str], int]],
    outputs: list,
    *,
    add_bos: bool,
    pooling: str,
    batch_size: int = 32,
    max_length: int | None = None,
    log_every: int = 50,
) -> dict:
    '''
    Run the model over the planned sentences and write pooled word vectors.

    This is the core of extract_features without any file handling, so it
    can be tested with a tiny model and plain NumPy arrays.

    Parameters
    ----------
    - model (PreTrainedModel): Frozen model with output_hidden_states=True.
    - tokenizer (PreTrainedTokenizerFast): Its tokenizer (with a pad token).
    - planned (list[tuple[str, list[str], int]]): Output of plan_sentences.
    - outputs (list): L + 1 writable arrays of shape [N, d] (memmaps or
    np.ndarray), one per layer, already filled with NaN.
    - add_bos (bool): True for decoders (see lib.alignment.encode_words).
    - pooling (str): "first", "mean" or "last".
    - batch_size (int): Sentences per forward pass. Defaults to 32.
    - max_length (int | None): Model length limit. Defaults to None.
    - log_every (int): Log progress every this many batches. Defaults to 50.

    Returns
    -------
    dict: {"n_truncated_sentences": int, "n_sentences": int, "seconds": float}.
    '''
    if pooling not in POOLINGS:
        raise ValueError(f"Unknown pooling '{pooling}'; expected one of {POOLINGS}.")
    device = next(model.parameters()).device
    backbone = getattr(model, "base_model", model)
    batches = make_batches([len(words) for _, words, _ in planned], batch_size)
    n_truncated = 0
    done = 0
    start = time.perf_counter()

    for batch_number, indices in enumerate(batches, start=1):
        sentences = [planned[i][1] for i in indices]
        batch, word_ids, truncated = encode_words_detailed(
            tokenizer, sentences, add_bos=add_bos, max_length=max_length
        )
        n_truncated += sum(truncated)
        with torch.inference_mode():
            # base_model skips the LM head of decoders: same hidden states, no [B, T, vocab] logits in memory.
            result = backbone(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                output_hidden_states=True,
            )
        hidden_states = result.hidden_states
        if len(hidden_states) != len(outputs):
            raise ValueError(f"The model returned {len(hidden_states)} hidden states, expected {len(outputs)}.")

        n_words = [len(words) for words in sentences]
        for layer, hidden in enumerate(hidden_states):
            pooled = pool_words(hidden, word_ids, n_words, pooling)
            stacked = torch.cat(pooled).cpu().numpy().astype(SAVED_DTYPE)
            if np.isinf(stacked).any():
                raise ValueError(
                    f"Float16 overflow at layer {layer} (batch {batch_number}); values exceed the float16 range."
                )
            position = 0
            for i, n in zip(indices, n_words):
                offset = planned[i][2]
                outputs[layer][offset: offset + n] = stacked[position: position + n]
                position += n

        done += len(indices)
        if batch_number % log_every == 0 or batch_number == len(batches):
            elapsed = time.perf_counter() - start
            logger.info(
                "%d/%d sentences, %.1f sentences/s", done, len(planned), done / max(elapsed, 1e-9)
            )

    return {
        "n_truncated_sentences": n_truncated,
        "n_sentences": len(planned),
        "seconds": round(time.perf_counter() - start, 1),
    }


def _git_commit() -> str | None:
    '''
    Short hash of the current commit of the repository, or None.

    Returns
    -------
    str | None: Output of "git rev-parse --short HEAD", or None if git is not
    available or the code does not live in a git repository.
    '''
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _count_nan_rows(array: np.ndarray, chunk: int = 50_000) -> int:
    '''
    Count rows that contain NaN, reading a memmap in chunks.

    Parameters
    ----------
    - array (np.ndarray): Array of shape [N, d].
    - chunk (int): Rows read at a time. Defaults to 50000.

    Returns
    -------
    int: Number of rows with at least one NaN.
    '''
    return int(sum(np.isnan(array[i: i + chunk]).any(axis=1).sum() for i in range(0, array.shape[0], chunk)))


def extract_features(
    model_key: str,
    treebank: str,
    split: str,
    *,
    init: str = "pretrained",
    pooling: str = "first",
    batch_size: int = 32,
    max_rows: int | None = None,
    overwrite: bool = False,
) -> Path:
    '''
    Extract and save word vectors of every layer for one treebank split.

    Parameters
    ----------
    - model_key (str): Registry key, for example "bert" or "gpt2".
    - treebank (str): Treebank key, for example "ewt" or "pud_en".
    - split (str): "train", "dev" or "test".
    - init (str): "pretrained" or "random" (random weights, seed 42, the same
    network for every team member). Defaults to "pretrained".
    - pooling (str): "first", "mean" or "last". Defaults to "first".
    - batch_size (int): Sentences per forward pass. Defaults to 32.
    - max_rows (int | None): Cap on rows, keeping whole sentences from the
    start of the word table (CONTRACTS.md section 4). Defaults to None.
    - overwrite (bool): Recompute even if the folder is complete. Defaults
    to False.

    Returns
    -------
    Path: The feature folder (CONTRACTS.md section 4), containing
    layer_00.npy ... layer_{L}.npy (float16, [N, d], NaN rows for words
    without subwords) and meta.json.
    '''
    if init not in INITS:
        raise ValueError(f"Unknown init '{init}'; expected one of {INITS}.")
    if pooling not in POOLINGS:
        raise ValueError(f"Unknown pooling '{pooling}'; expected one of {POOLINGS}.")

    folder = feature_dir(model_key, init, pooling, treebank, split)
    meta_path = folder / "meta.json"
    if meta_path.is_file() and not overwrite:
        logger.info("Already complete, skipping (use overwrite=True to recompute): %s", folder)
        return folder

    spec = get_model_spec(model_key)
    is_decoder = spec["kind"] == "decoder"
    device = default_device()
    dtype = spec["dtype"]
    if device == "cpu" and dtype == "float16":
        dtype = "bfloat16"  # float16 is slow or unsupported on CPU
        logger.warning("Loading %s in bfloat16 instead of float16 because the device is CPU.", model_key)

    model, tokenizer, model_meta = load_model(
        spec["hf_id"],
        kind=spec["kind"],
        random_weights=(init == "random"),
        seed=SEED,
        device=device,
        revision=spec.get("revision", "main"),
        dtype=dtype,
        tokenizer_kwargs={"add_prefix_space": True} if is_decoder else None,
    )
    num_layers, hidden_size = model_meta["num_layers"], model_meta["hidden_size"]
    if (num_layers, hidden_size) != (spec["num_layers"], spec["hidden_size"]):
        raise ValueError(
            f"{model_key}: the registry says L={spec['num_layers']}, d={spec['hidden_size']}, "
            f"the model has L={num_layers}, d={hidden_size}."
        )
    max_length = model_max_length(model.config)

    table = load_word_table(treebank, split)
    sentences = table_to_sentences(table)
    if sum(len(words) for _, words in sentences) != len(table):
        raise ValueError("table_to_sentences does not cover every row of the word table.")
    planned, n_rows = plan_sentences(sentences, max_rows)

    folder.mkdir(parents=True, exist_ok=True)
    meta_path.unlink(missing_ok=True)  # the folder stays "incomplete" until the new meta.json is written
    outputs = []
    for layer in range(num_layers + 1):
        path = layer_path(model_key, init, pooling, treebank, split, layer)
        try:
            outputs.append(open_memmap(path, mode="w+", dtype=SAVED_DTYPE, shape=(n_rows, hidden_size)))
        except OSError as error:
            raise OSError(
                f"Cannot write {path}. On Windows a .npy file cannot be overwritten while it is open: "
                "close notebooks or scripts that loaded these features (load_layer) and run again."
            ) from error
        outputs[-1][:] = np.nan

    logger.info(
        "Extracting %s (%s, %s) on %s/%s: %d sentences, %d rows, %d layers, device %s",
        model_key, init, pooling, treebank, split, len(planned), n_rows, num_layers + 1, device,
    )
    stats = extract_into(
        model, tokenizer, planned, outputs,
        add_bos=is_decoder, pooling=pooling, batch_size=batch_size, max_length=max_length,
    )
    for array in outputs:
        array.flush()
    n_nan_rows = _count_nan_rows(outputs[0])
    del outputs  # release the memory maps (Windows keeps mapped files locked)

    meta = {
        "model": model_key,
        "hf_id": spec["hf_id"],
        "init": init,
        "pooling": pooling,
        "treebank": treebank,
        "split": split,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "n_rows": n_rows,
        "n_nan_rows": n_nan_rows,
        "n_truncated_sentences": stats["n_truncated_sentences"],
        "dtype_model": model_meta["dtype"].replace("torch.", ""),
        "dtype_saved": "float16",
        "seed": SEED,
        "max_rows": max_rows,
        "git_commit": _git_commit(),
        "created": datetime.now().isoformat(timespec="seconds"),
        # Extra fields (not required by the contract, safe to ignore):
        "n_sentences": stats["n_sentences"],
        "max_length": max_length,
        "batch_size": batch_size,
        "device": device,
        "seconds": stats["seconds"],
        # Decoders: Hugging Face applies the final layer norm (ln_f) to hidden_states[L] only,
        # so the last layer has a different scale from layers 0..L-1 (CONTRACTS.md section 9.2).
        "final_norm_on_last_layer": is_decoder,
    }
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    logger.info("Done in %.1f s: %s (%d NaN rows)", stats["seconds"], folder, n_nan_rows)
    return folder


def load_meta(model: str, init: str, pooling: str, treebank: str, split: str) -> dict:
    '''
    Read meta.json of a feature folder.

    Parameters
    ----------
    - model, init, pooling, treebank, split (str): Feature folder keys
    (CONTRACTS.md section 4).

    Returns
    -------
    dict: The meta.json content. Raises FileNotFoundError if the folder is
    missing or incomplete.
    '''
    path = feature_dir(model, init, pooling, treebank, split) / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No meta.json in {path.parent}: the extraction is missing or incomplete. "
            f"Run scripts/01_extract_features.py --model {model} --treebank {treebank} --split {split} "
            f"--init {init} --pooling {pooling} --overwrite"
        )
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_layer(model: str, init: str, pooling: str, treebank: str, split: str, layer: int) -> np.ndarray:
    '''
    Open the features of one layer as a read-only memory map.

    Parameters
    ----------
    - model, init, pooling, treebank, split (str): Feature folder keys
    (CONTRACTS.md section 4).
    - layer (int): Layer index, 0 (embeddings) to L.

    Returns
    -------
    np.ndarray: float16 array of shape [N, d], aligned with the word table
    rows; NaN rows for words without subwords (drop them, and the same rows
    of the labels, before fitting). Cast to float32 before training.
    '''
    meta = load_meta(model, init, pooling, treebank, split)  # refuses incomplete folders
    if not 0 <= layer <= meta["num_layers"]:
        raise ValueError(f"layer must be in [0, {meta['num_layers']}], got {layer}.")
    return np.load(layer_path(model, init, pooling, treebank, split, layer), mmap_mode="r")
