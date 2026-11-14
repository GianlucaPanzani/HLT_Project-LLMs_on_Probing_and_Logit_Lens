"""Tests for lib.features (issue E2, #4)."""

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import lib.features as features  # noqa: E402
from lib.features import (  # noqa: E402
    extract_features,
    extract_into,
    load_layer,
    load_meta,
    make_batches,
    plan_sentences,
)

CONTRACT_META_KEYS = {
    "model", "hf_id", "init", "pooling", "treebank", "split", "num_layers", "hidden_size", "n_rows",
    "n_nan_rows", "n_truncated_sentences", "dtype_model", "dtype_saved", "seed", "max_rows", "git_commit", "created",
}

# Different lengths on purpose, so that sorting by length really reorders them.
SENTENCES = [
    ("s0", ["The", "cat", "."]),
    ("s1", ["New", "York", "'s", "skyscrapers", "are", "on", "the", "mat", "."]),
    ("s2", ["word"]),
    ("s3", ["The", "​", "cat", "sat"]),  # zero-width space: zero subwords, NaN row
    ("s4", ["Il", "gatto", "dell'", "anno", "è", "bello", "."]),
    ("s5", ["a", "b"]),
]
N_ROWS = sum(len(w) for _, w in SENTENCES)


def _tiny_bert(seed: int):
    torch.manual_seed(seed)
    config = transformers.BertConfig(
        vocab_size=200, hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
        output_hidden_states=True,
    )
    return transformers.BertModel(config).eval()


def _tiny_gpt2(seed: int):
    torch.manual_seed(seed)
    config = transformers.GPT2Config(
        vocab_size=512, n_embd=16, n_layer=2, n_head=2, n_positions=64, output_hidden_states=True, use_cache=False,
    )
    return transformers.GPT2LMHeadModel(config).eval()


def _run(model, tokenizer, add_bos, pooling, batch_size, dim=16, layers=3):
    planned, n_rows = plan_sentences(SENTENCES)
    outputs = [np.full((n_rows, dim), np.nan, dtype=np.float16) for _ in range(layers)]
    stats = extract_into(model, tokenizer, planned, outputs, add_bos=add_bos, pooling=pooling,
                         batch_size=batch_size, max_length=64)
    return outputs, stats


# ---------------------------------------------------------------- pure helpers

def test_plan_sentences_offsets_and_max_rows():
    planned, n_rows = plan_sentences(SENTENCES)
    assert [p[2] for p in planned] == [0, 3, 12, 13, 17, 24] and n_rows == N_ROWS
    planned, n_rows = plan_sentences(SENTENCES, max_rows=13)  # whole sentences only
    assert [p[0] for p in planned] == ["s0", "s1", "s2"] and n_rows == 13
    with pytest.raises(ValueError):
        plan_sentences(SENTENCES, max_rows=0)


def test_make_batches_sorted_by_length():
    assert make_batches([3, 9, 1, 4, 7, 2], 2) == [[1, 4], [3, 0], [5, 2]]


# ---------------------------------------------------------------- rows land at the right offsets

@pytest.mark.parametrize("pooling", ["first", "last", "mean"])
def test_sorted_batches_match_one_sentence_at_a_time_encoder(bert_tokenizer, pooling):
    model = _tiny_bert(0)
    batched, stats = _run(model, bert_tokenizer, False, pooling, batch_size=4)
    single, _ = _run(model, bert_tokenizer, False, pooling, batch_size=1)
    assert stats["n_sentences"] == len(SENTENCES) and stats["n_truncated_sentences"] == 0
    for a, b in zip(batched, single):
        np.testing.assert_allclose(a.astype(np.float32), b.astype(np.float32), atol=2e-3, equal_nan=True)
    nan_rows = np.isnan(batched[0]).any(axis=1)
    assert nan_rows.tolist() == [i == 14 for i in range(N_ROWS)]  # only the zero-width word of s3 (row 13 + 1)

    # Each sentence encoded alone and pooled directly gives exactly its own rows.
    from lib.alignment import encode_words, pool_words

    for _, words, offset in plan_sentences(SENTENCES)[0]:
        batch, word_ids = encode_words(bert_tokenizer, [words], add_bos=False, max_length=64)
        with torch.inference_mode():
            hidden = model(**batch).hidden_states[2]
        expected = pool_words(hidden, word_ids, [len(words)], pooling)[0].numpy()
        np.testing.assert_allclose(
            batched[2][offset: offset + len(words)].astype(np.float32), expected, atol=2e-2, rtol=1e-2, equal_nan=True
        )


@pytest.mark.parametrize("pooling", ["first", "last"])
def test_sorted_batches_match_one_sentence_at_a_time_decoder(gpt2_tokenizer, pooling):
    model = _tiny_gpt2(0)
    batched, _ = _run(model, gpt2_tokenizer, True, pooling, batch_size=4)
    single, _ = _run(model, gpt2_tokenizer, True, pooling, batch_size=1)
    for a, b in zip(batched, single):
        np.testing.assert_allclose(a.astype(np.float32), b.astype(np.float32), atol=2e-3, equal_nan=True)


def test_decoder_backbone_gives_the_same_hidden_states_as_the_lm(gpt2_tokenizer):
    from lib.alignment import encode_words

    model = _tiny_gpt2(0)
    batch, _ = encode_words(gpt2_tokenizer, [w for _, w in SENTENCES], add_bos=True, max_length=64)
    with torch.inference_mode():
        full = model(**batch, output_hidden_states=True).hidden_states
        base = model.base_model(**batch, output_hidden_states=True).hidden_states
    assert len(full) == len(base) == 3
    for a, b in zip(full, base):
        torch.testing.assert_close(a, b)


def test_wrong_number_of_layers_is_detected(bert_tokenizer):
    with pytest.raises(ValueError, match="hidden states"):
        _run(_tiny_bert(0), bert_tokenizer, False, "first", batch_size=4, layers=5)


# ---------------------------------------------------------------- extract_features end to end (no download)

@pytest.fixture
def fake_env(monkeypatch, tmp_path, bert_tokenizer):
    """Replace the registry, model loader, word table and paths with tiny local versions."""
    calls = {"load_model": 0}

    def fake_spec(key):
        return {"key": key, "hf_id": "tiny-bert", "kind": "encoder", "num_layers": 2, "hidden_size": 16,
                "dtype": "float32"}

    def fake_load_model(checkpoint, *, kind, random_weights=False, seed=42, device=None, dtype=None,
                        tokenizer_kwargs=None, revision="main"):
        calls["load_model"] += 1
        model = _tiny_bert(seed if random_weights else 0)
        return model, bert_tokenizer, {"num_layers": 2, "hidden_size": 16, "dtype": "torch.float32"}

    table = pd.DataFrame([
        {"sent_id": sid, "word_idx": i, "form": w, "upos": "X", "next_upos": None, "lang": "en"}
        for sid, words in SENTENCES for i, w in enumerate(words)
    ])

    def fake_table_to_sentences(t):
        return [(sid, list(g["form"])) for sid, g in t.groupby("sent_id", sort=False)]

    def fake_dir(model, init, pooling, treebank, split):
        return tmp_path / model / init / pooling / f"{treebank}_{split}"

    monkeypatch.setattr(features, "get_model_spec", fake_spec)
    monkeypatch.setattr(features, "load_model", fake_load_model)
    monkeypatch.setattr(features, "load_word_table", lambda tb, split: table)
    monkeypatch.setattr(features, "table_to_sentences", fake_table_to_sentences)
    monkeypatch.setattr(features, "feature_dir", fake_dir)
    monkeypatch.setattr(features, "layer_path", lambda *a: fake_dir(*a[:5]) / f"layer_{a[5]:02d}.npy")
    monkeypatch.setattr(features, "default_device", lambda: "cpu")
    return calls


def test_extract_features_end_to_end(fake_env):
    folder = extract_features("tiny", "ewt", "test", batch_size=4)
    meta = json.loads((folder / "meta.json").read_text())
    assert CONTRACT_META_KEYS <= set(meta)
    assert meta["n_rows"] == N_ROWS and meta["n_nan_rows"] == 1 and meta["dtype_saved"] == "float16"
    assert meta["dtype_model"] == "float32" and meta["seed"] == 42 and meta["max_rows"] is None
    assert load_meta("tiny", "pretrained", "first", "ewt", "test") == meta
    for layer in range(3):
        array = load_layer("tiny", "pretrained", "first", "ewt", "test", layer)
        assert array.shape == (N_ROWS, 16) and array.dtype == np.float16
    del array  # Windows cannot overwrite a file that is still memory-mapped
    with pytest.raises(ValueError):
        load_layer("tiny", "pretrained", "first", "ewt", "test", 3)

    # A complete folder is skipped (the model is not even loaded).
    extract_features("tiny", "ewt", "test")
    assert fake_env["load_model"] == 1

    # Overwrite gives identical arrays; random init gives different ones.
    before = np.array(load_layer("tiny", "pretrained", "first", "ewt", "test", 2))
    extract_features("tiny", "ewt", "test", overwrite=True)
    after = np.array(load_layer("tiny", "pretrained", "first", "ewt", "test", 2))
    np.testing.assert_array_equal(before, after)
    extract_features("tiny", "ewt", "test", init="random")
    rand = np.array(load_layer("tiny", "random", "first", "ewt", "test", 2))
    assert not np.allclose(np.nan_to_num(before), np.nan_to_num(rand))


def test_extract_features_max_rows(fake_env):
    folder = extract_features("tiny", "ewt", "test", max_rows=13)
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["n_rows"] == 13 and meta["max_rows"] == 13 and meta["n_nan_rows"] == 0
    assert load_layer("tiny", "pretrained", "first", "ewt", "test", 0).shape == (13, 16)


def test_open_memory_map_gives_a_clear_error(fake_env, monkeypatch):
    def locked(*args, **kwargs):
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(features, "open_memmap", locked)
    with pytest.raises(OSError, match="close notebooks"):
        extract_features("tiny", "ewt", "test")


def test_incomplete_folder_is_refused(fake_env):
    folder = extract_features("tiny", "ewt", "test")
    (folder / "meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        load_layer("tiny", "pretrained", "first", "ewt", "test", 0)


def test_bad_arguments():
    with pytest.raises(ValueError, match="init"):
        extract_features("bert", "ewt", "test", init="trained")
    with pytest.raises(ValueError, match="pooling"):
        extract_features("bert", "ewt", "test", pooling="max")


# ---------------------------------------------------------------- real model and data (not in CI)

@pytest.mark.slow
def test_real_bert_on_ewt_test(tmp_path, monkeypatch):
    monkeypatch.setenv("HLT_FEATURES_DIR", str(tmp_path))
    from lib.word_table import load_word_table

    folder = extract_features("bert", "ewt", "test", max_rows=500)
    meta = load_meta("bert", "pretrained", "first", "ewt", "test")
    assert meta["n_rows"] <= 500 and folder.is_dir()
    layers = {layer: np.array(load_layer("bert", "pretrained", "first", "ewt", "test", layer)) for layer in (0, 12)}
    for array in layers.values():
        assert array.shape == (meta["n_rows"], 768)
        assert int(np.isnan(array).any(axis=1).sum()) == meta["n_nan_rows"]
    assert meta["n_rows"] <= len(load_word_table("ewt", "test"))

    extract_features("bert", "ewt", "test", max_rows=500, overwrite=True)
    np.testing.assert_array_equal(layers[12], np.array(load_layer("bert", "pretrained", "first", "ewt", "test", 12)))
    extract_features("bert", "ewt", "test", max_rows=500, init="random")
    rand = np.array(load_layer("bert", "random", "first", "ewt", "test", 12))
    assert not np.allclose(np.nan_to_num(layers[12]), np.nan_to_num(rand))
