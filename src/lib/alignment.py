"""Subword-to-word alignment (CONTRACTS.md sections 3 and 10).

Turns pre-split UD words into model inputs and pools subword hidden states
back to exactly one vector per UD word, so that every feature row matches a
row of the word table.
"""

import logging

import torch
from transformers import BatchEncoding

logger = logging.getLogger(__name__)

POOLINGS: tuple[str, ...] = ("first", "mean", "last")


def _check_tokenizer(tokenizer, add_bos: bool) -> None:
    '''
    Check that a tokenizer can be used for word alignment.

    Parameters
    ----------
    - tokenizer (PreTrainedTokenizerFast): Tokenizer to check.
    - add_bos (bool): True for decoders, which need a BOS (or EOS) token and,
    for byte-level BPE tokenizers, add_prefix_space=True.

    Returns
    -------
    None. Raises ValueError when the tokenizer is unusable.
    '''
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for word alignment (encoding.word_ids()).")
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer has no pad token; load it with lib.data.load_model, which sets one.")
    if add_bos:
        if tokenizer.bos_token_id is None and tokenizer.eos_token_id is None:
            raise ValueError("add_bos=True needs a tokenizer with a BOS or EOS token.")
        if getattr(tokenizer, "add_prefix_space", None) is False:
            raise ValueError(
                "GPT-2-family tokenizers need add_prefix_space=True with pre-split words; "
                "pass tokenizer_kwargs={'add_prefix_space': True} to load_model (CONTRACTS.md section 1.1)."
            )


def _encode_sentence(
    tokenizer,
    words: list[str],
    add_bos: bool,
    max_length: int | None,
) -> tuple[list[int], list[int | None], bool]:
    '''
    Encode one pre-split sentence.

    Parameters
    ----------
    - tokenizer (PreTrainedTokenizerFast): Fast tokenizer.
    - words (list[str]): UD word forms of one sentence. Must not be empty.
    - add_bos (bool): If True (decoders), tokenize without special tokens and
    prepend the BOS token (EOS if there is no BOS) with word id None. If
    False (encoders), add the tokenizer's own special tokens.
    - max_length (int | None): Maximum sequence length including special
    tokens. None means no truncation.

    Returns
    -------
    A tuple (input_ids, word_ids, truncated): token ids, the word index of
    each position (None for special tokens), and whether the sentence was
    cut to max_length.
    '''
    if not words:
        raise ValueError("Cannot encode an empty sentence.")

    if add_bos:
        encoding = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
        bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        input_ids = [bos] + list(encoding["input_ids"])
        word_ids = [None] + list(encoding.word_ids())
        truncated = max_length is not None and len(input_ids) > max_length
        if truncated:
            input_ids, word_ids = input_ids[:max_length], word_ids[:max_length]
        return input_ids, word_ids, truncated

    # Encoders: tokenizer truncation, as required by the issue. A second, untruncated
    # call is only needed to know whether truncation happened (rare in UD).
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        truncation=max_length is not None,
        max_length=max_length,
    )
    input_ids = list(encoding["input_ids"])
    truncated = False
    if max_length is not None and len(input_ids) >= max_length:
        full = tokenizer(words, is_split_into_words=True, add_special_tokens=True)
        truncated = len(full["input_ids"]) > max_length
    return input_ids, list(encoding.word_ids()), truncated


def encode_words_detailed(
    tokenizer,
    sentences: list[list[str]],
    *,
    add_bos: bool,
    max_length: int | None = None,
) -> tuple[BatchEncoding, list[list[int | None]], list[bool]]:
    '''
    Same as encode_words, and also report which sentences were truncated.
    Used by lib.features to fill n_truncated_sentences in meta.json.

    Parameters
    ----------
    - tokenizer (PreTrainedTokenizerFast): Fast tokenizer with a pad token
    (lib.data.load_model sets one).
    - sentences (list[list[str]]): Batch of pre-split sentences (UD word forms).
    - add_bos (bool): True for decoders, False for encoders (see encode_words).
    - max_length (int | None): Model length limit (CONTRACTS.md section 3).
    Do not pass tokenizer.model_max_length, which can be a huge sentinel.

    Returns
    -------
    A tuple (batch, word_ids, truncated): the padded BatchEncoding, the
    per-sentence word id lists padded with None to the batch length, and one
    flag per sentence telling whether it was truncated.
    '''
    _check_tokenizer(tokenizer, add_bos)
    if not sentences:
        raise ValueError("sentences must not be empty.")
    if max_length is not None and max_length < 2:
        raise ValueError("max_length must be at least 2 (one special token plus one subword).")

    encoded = [_encode_sentence(tokenizer, words, add_bos, max_length) for words in sentences]
    length = max(len(ids) for ids, _, _ in encoded)

    input_ids = torch.full((len(encoded), length), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), length), dtype=torch.long)
    word_ids: list[list[int | None]] = []
    truncated: list[bool] = []
    for row, (ids, wids, was_truncated) in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1
        word_ids.append(wids + [None] * (length - len(wids)))
        truncated.append(was_truncated)

    n_dropped = sum(
        count_missing_words(wids, len(words)) for wids, words, cut in zip(word_ids, sentences, truncated) if cut
    )
    if any(truncated):
        logger.warning(
            "Truncated %d sentence(s) to max_length=%s; %d word(s) lost all their subwords and get NaN rows.",
            sum(truncated), max_length, n_dropped,
        )

    batch = BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})
    return batch, word_ids, truncated


def encode_words(
    tokenizer,
    sentences: list[list[str]],
    *,
    add_bos: bool,
    max_length: int | None = None,
) -> tuple[BatchEncoding, list[list[int | None]]]:
    '''
    Tokenize pre-split sentences and keep the subword-to-word mapping.

    Parameters
    ----------
    - tokenizer (PreTrainedTokenizerFast): Fast tokenizer with a pad token
    (lib.data.load_model sets one). GPT-2-family tokenizers must be loaded
    with add_prefix_space=True.
    - sentences (list[list[str]]): Batch of pre-split sentences (UD word forms).
    - add_bos (bool): True for decoders: no special tokens are added by the
    tokenizer and the BOS token (EOS if there is no BOS) is prepended with
    word id None, so the first word has a context position. False for
    encoders: the tokenizer adds its own special tokens ([CLS], [SEP]).
    - max_length (int | None): Model length limit including special tokens
    (512 for BERT-family, 1024 for GPT-2, 2048 for mGPT). Longer sequences
    are truncated; words that lose all their subwords get NaN rows in
    pool_words. None means no truncation. Defaults to None.

    Returns
    -------
    A tuple (batch, word_ids):
    - batch (BatchEncoding): input_ids and attention_mask, torch.long tensors
    of shape [B, T], right-padded with tokenizer.pad_token_id.
    - word_ids (list[list[int | None]]): For each sentence, T entries giving
    the 0-based word index of each position; None for special tokens, BOS
    and padding.
    '''
    batch, word_ids, _ = encode_words_detailed(tokenizer, sentences, add_bos=add_bos, max_length=max_length)
    return batch, word_ids


def count_missing_words(word_ids: list[int | None], n_words: int) -> int:
    '''
    Count the words of one sentence that have no subword position.

    Parameters
    ----------
    - word_ids (list[int | None]): Word id per position for one sentence.
    - n_words (int): Number of UD words in the sentence.

    Returns
    -------
    int: Number of word indices in [0, n_words) that never appear in word_ids
    (words removed by truncation or that tokenize to zero subwords).
    '''
    present = {w for w in word_ids if w is not None}
    return n_words - len(present & set(range(n_words)))


def pool_words(
    hidden: torch.Tensor,
    word_ids: list[list[int | None]],
    n_words: list[int],
    pooling: str,
) -> list[torch.Tensor]:
    '''
    Pool subword hidden states of one layer into one vector per UD word.

    Parameters
    ----------
    - hidden (torch.Tensor): Hidden states of one layer, shape [B, T, d].
    - word_ids (list[list[int | None]]): Per sentence, the word id of each
    position as returned by encode_words (at most T entries; None entries,
    such as special tokens, BOS and padding, are ignored).
    - n_words (list[int]): Number of UD words of each sentence.
    - pooling (str): "first" (first subword of the word, default for the main
    results), "last" (last subword; needed by decoders for upos_next and the
    lens bridge) or "mean" (mean over the word's subwords, ablation).

    Returns
    -------
    list[torch.Tensor]: One float32 tensor of shape [n_words_i, d] per
    sentence, on the device of hidden. Words without any position
    (truncated, or zero subwords) are rows of NaN, so the following words
    never shift.
    '''
    if pooling not in POOLINGS:
        raise ValueError(f"Unknown pooling '{pooling}'; expected one of {POOLINGS}.")
    if hidden.dim() != 3:
        raise ValueError(f"hidden must have shape [B, T, d], got {tuple(hidden.shape)}.")
    if not (len(word_ids) == len(n_words) == hidden.shape[0]):
        raise ValueError(
            f"Batch size mismatch: hidden has {hidden.shape[0]} rows, "
            f"word_ids has {len(word_ids)}, n_words has {len(n_words)}."
        )

    hidden = hidden.float()
    _, length, dim = hidden.shape
    pooled = []
    for row, (wids, n) in enumerate(zip(word_ids, n_words)):
        if len(wids) > length:
            raise ValueError(f"Sentence {row} has {len(wids)} word ids but hidden has only {length} positions.")
        positions = [t for t, w in enumerate(wids) if w is not None]
        words = [wids[t] for t in positions]
        if any(w < 0 or w >= n for w in words):
            raise ValueError(f"Sentence {row} has a word id outside [0, {n}).")

        out = torch.full((n, dim), float("nan"), dtype=torch.float32, device=hidden.device)
        if not positions:
            pooled.append(out)
            continue

        pos = torch.tensor(positions, dtype=torch.long, device=hidden.device)
        wid = torch.tensor(words, dtype=torch.long, device=hidden.device)
        states = hidden[row, pos]  # [n_positions, d]

        if pooling == "mean":
            sums = torch.zeros((n, dim), dtype=torch.float32, device=hidden.device).index_add_(0, wid, states)
            counts = torch.bincount(wid, minlength=n)
            seen = counts > 0
            out[seen] = sums[seen] / counts[seen].unsqueeze(1).float()
        else:
            # Positions are in increasing order: for "first" keep the first
            # occurrence of each word, for "last" the last one.
            order = range(len(words)) if pooling == "last" else range(len(words) - 1, -1, -1)
            chosen = {}
            for k in order:
                chosen[words[k]] = k  # later assignments win
            idx_words = torch.tensor(list(chosen.keys()), dtype=torch.long, device=hidden.device)
            idx_states = torch.tensor(list(chosen.values()), dtype=torch.long, device=hidden.device)
            out[idx_words] = states[idx_states]
        pooled.append(out)
    return pooled
