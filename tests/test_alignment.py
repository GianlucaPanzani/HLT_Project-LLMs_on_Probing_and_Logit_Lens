"""Tests for lib.alignment (issue E1, #3)."""

import logging

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from lib.alignment import count_missing_words, encode_words, encode_words_detailed, pool_words  # noqa: E402

SENTENCES = [
    ["New", "York", "'s", "skyscrapers"],
    ["The", "cat", "sat", "on", "the", "mat", "."],
    ["Il", "gatto", "dell'", "anno", "è", "bello", "."],
]


def _fake_hidden(batch: int, length: int, dim: int = 3) -> torch.Tensor:
    """hidden[b, t, :] == 100 * b + t, so every position is recognisable."""
    values = torch.arange(batch)[:, None] * 100 + torch.arange(length)[None, :]
    return values[:, :, None].repeat(1, 1, dim).float()


# ---------------------------------------------------------------- pool_words (synthetic)

def test_pool_first_last_mean_known_values():
    hidden = _fake_hidden(1, 9)
    # positions: 0=[CLS], 1-2 word 0, 3 word 1, 4-6 word 2, word 3 missing, 7=[SEP], 8=pad
    word_ids = [[None, 0, 0, 1, 2, 2, 2, None, None]]
    first, = pool_words(hidden, word_ids, [4], "first")
    last, = pool_words(hidden, word_ids, [4], "last")
    mean, = pool_words(hidden, word_ids, [4], "mean")
    assert first[:3, 0].tolist() == [1.0, 3.0, 4.0]
    assert last[:3, 0].tolist() == [2.0, 3.0, 6.0]
    assert mean[:3, 0].tolist() == [1.5, 3.0, 5.0]
    for out in (first, last, mean):
        assert out.shape == (4, 3)
        assert torch.isnan(out[3]).all()
        assert not torch.isnan(out[:3]).any()


def test_missing_word_in_the_middle_does_not_shift_following_words():
    hidden = _fake_hidden(1, 6)
    word_ids = [[None, 0, 2, 3, 3, None]]  # word 1 has zero subwords
    for pooling in ("first", "last", "mean"):
        out, = pool_words(hidden, word_ids, [4], pooling)
        assert out[0, 0] == 1.0
        assert torch.isnan(out[1]).all()
        assert out[2, 0] == 2.0


def test_shapes_and_dtype_for_a_batch():
    hidden = _fake_hidden(2, 5).half()
    word_ids = [[None, 0, 1, None, None], [None, 0, 1, 2, None]]
    outs = pool_words(hidden, word_ids, [2, 3], "first")
    assert [tuple(o.shape) for o in outs] == [(2, 3), (3, 3)]
    assert all(o.dtype == torch.float32 for o in outs)
    assert outs[1][2, 0] == 103.0  # sentence 1, position 3


def test_word_ids_shorter_than_hidden_are_accepted():
    hidden = _fake_hidden(1, 6)
    out, = pool_words(hidden, [[None, 0, 1]], [2], "last")
    assert out[:, 0].tolist() == [1.0, 2.0]


def test_errors():
    hidden = _fake_hidden(1, 4)
    with pytest.raises(ValueError, match="Unknown pooling"):
        pool_words(hidden, [[None, 0, None, None]], [1], "max")
    with pytest.raises(ValueError, match="outside"):
        pool_words(hidden, [[None, 0, 5, None]], [2], "first")
    with pytest.raises(ValueError, match="mismatch"):
        pool_words(hidden, [[None, 0], [None, 0]], [1, 1], "first")


def test_count_missing_words():
    assert count_missing_words([None, 0, 0, 2, None], 4) == 2


# ---------------------------------------------------------------- encode_words (local tokenizers)

def _check_alignment(batch, word_ids, sentences):
    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long
    length = batch["input_ids"].shape[1]
    for row, (wids, words) in enumerate(zip(word_ids, sentences)):
        assert len(wids) == length
        real = int(batch["attention_mask"][row].sum())
        assert all(w is None for w in wids[real:])  # padding has no word id
        assert {w for w in wids if w is not None} == set(range(len(words)))  # every word has a position


def test_encoder_encoding(bert_tokenizer):
    batch, word_ids = encode_words(bert_tokenizer, SENTENCES, add_bos=False, max_length=512)
    _check_alignment(batch, word_ids, SENTENCES)
    for row, wids in enumerate(word_ids):
        real = int(batch["attention_mask"][row].sum())
        assert batch["input_ids"][row, 0] == bert_tokenizer.cls_token_id and wids[0] is None
        assert batch["input_ids"][row, real - 1] == bert_tokenizer.sep_token_id and wids[real - 1] is None
        assert (batch["input_ids"][row, real:] == bert_tokenizer.pad_token_id).all()
    pooled = pool_words(torch.randn(*batch["input_ids"].shape, 4), word_ids, [len(s) for s in SENTENCES], "first")
    assert [o.shape[0] for o in pooled] == [len(s) for s in SENTENCES]


def test_decoder_encoding_has_bos_only_at_start(gpt2_tokenizer):
    batch, word_ids = encode_words(gpt2_tokenizer, SENTENCES, add_bos=True, max_length=1024)
    _check_alignment(batch, word_ids, SENTENCES)
    bos = gpt2_tokenizer.bos_token_id
    for row, wids in enumerate(word_ids):
        assert batch["input_ids"][row, 0] == bos and wids[0] is None
        real = int(batch["attention_mask"][row].sum())
        assert all(w is not None for w in wids[1:real])  # no other special token


def test_decoder_requires_add_prefix_space(gpt2_tokenizer):
    gpt2_tokenizer.add_prefix_space = False
    try:
        with pytest.raises(ValueError, match="add_prefix_space"):
            encode_words(gpt2_tokenizer, SENTENCES, add_bos=True)
    finally:
        gpt2_tokenizer.add_prefix_space = True


def test_slow_tokenizer_is_rejected():
    class Slow:
        is_fast = False
        pad_token_id = 0

    with pytest.raises(ValueError, match="fast tokenizer"):
        encode_words(Slow(), SENTENCES, add_bos=False)


def test_zero_subword_word_gets_nan_row(bert_tokenizer):
    sentence = ["The", "​", "cat"]  # zero-width space: removed by the BERT normalizer
    batch, word_ids = encode_words(bert_tokenizer, [sentence], add_bos=False, max_length=512)
    assert 1 not in word_ids[0]
    hidden = _fake_hidden(1, batch["input_ids"].shape[1])
    out, = pool_words(hidden, word_ids, [3], "first")
    assert out[0, 0] == 1.0 and torch.isnan(out[1]).all() and out[2, 0] == 2.0


def test_encoder_truncation_gives_nan_rows_only_at_the_end(bert_tokenizer, caplog):
    sentence = [f"w{i % 10}" for i in range(600)]  # one subword per word
    with caplog.at_level(logging.WARNING, logger="lib.alignment"):
        batch, word_ids, truncated = encode_words_detailed(bert_tokenizer, [sentence], add_bos=False, max_length=512)
    assert batch["input_ids"].shape[1] == 512 and truncated == [True]
    out, = pool_words(torch.randn(1, 512, 2), word_ids, [600], "first")
    nan_rows = torch.isnan(out).all(dim=1)
    assert not nan_rows[:510].any() and nan_rows[510:].all()  # 512 minus [CLS] and [SEP]
    assert "90 word(s)" in caplog.text


def test_decoder_truncation(gpt2_tokenizer):
    sentence = ["word"] * 50
    batch, word_ids, truncated = encode_words_detailed(gpt2_tokenizer, [sentence], add_bos=True, max_length=20)
    assert batch["input_ids"].shape[1] == 20 and truncated == [True] and word_ids[0][0] is None
    _, _, not_cut = encode_words_detailed(gpt2_tokenizer, [["word"] * 3], add_bos=True, max_length=20)
    assert not_cut == [False]


def test_empty_sentence_is_rejected(bert_tokenizer):
    with pytest.raises(ValueError, match="empty"):
        encode_words(bert_tokenizer, [[]], add_bos=False)


# ---------------------------------------------------------------- real tokenizers (download, not in CI)

@pytest.fixture(scope="module")
def real_tokenizers():
    from transformers import AutoTokenizer

    bert = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)
    gpt2 = AutoTokenizer.from_pretrained("gpt2", use_fast=True, add_prefix_space=True)
    gpt2.pad_token = gpt2.eos_token
    return bert, gpt2


@pytest.mark.slow
def test_real_tokenizers_alignment(real_tokenizers):
    bert, gpt2 = real_tokenizers
    assert gpt2.add_prefix_space  # CONTRACTS.md section 1.1
    for tokenizer, add_bos in ((bert, False), (gpt2, True)):
        batch, word_ids = encode_words(tokenizer, SENTENCES, add_bos=add_bos, max_length=512)
        _check_alignment(batch, word_ids, SENTENCES)
        pooled = pool_words(torch.zeros(*batch["input_ids"].shape, 2), word_ids, [len(s) for s in SENTENCES], "last")
        assert [o.shape[0] for o in pooled] == [len(s) for s in SENTENCES]
        first_ids = batch["input_ids"][:, 0].tolist()
        if add_bos:
            assert all(i == gpt2.bos_token_id for i in first_ids)
            assert all(w[0] is None for w in word_ids)
        else:
            assert gpt2.bos_token_id not in batch["input_ids"].flatten().tolist()
            assert all(i == bert.cls_token_id for i in first_ids)


@pytest.mark.slow
def test_real_bert_600_words(real_tokenizers, caplog):
    bert, _ = real_tokenizers
    sentence = ["word"] * 600
    with caplog.at_level(logging.WARNING, logger="lib.alignment"):
        batch, word_ids, truncated = encode_words_detailed(bert, [sentence], add_bos=False, max_length=512)
    assert truncated == [True] and batch["input_ids"].shape[1] == 512
    out, = pool_words(torch.randn(1, 512, 2), word_ids, [600], "first")
    nan_rows = torch.isnan(out).all(dim=1).tolist()
    n_nan = sum(nan_rows)
    assert n_nan > 0 and not any(nan_rows[: 600 - n_nan]) and all(nan_rows[600 - n_nan:])
    assert f"{n_nan} word(s)" in caplog.text
    print(f"600-word sentence: {n_nan} words dropped")
