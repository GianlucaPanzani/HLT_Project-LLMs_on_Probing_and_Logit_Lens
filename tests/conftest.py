"""Shared test fixtures: tiny tokenizers built locally, so no test downloads anything."""

import pytest

WORDS = [
    "The", "the", "cat", "sat", "on", "mat", ".", ",", "New", "York", "'s", "skyscrapers",
    "Il", "gatto", "dell'", "anno", "è", "bello", "word", "_", "a", "b", "c",
]


def _corpus() -> list[str]:
    return [
        "The cat sat on the mat .",
        "New York 's skyscrapers are tall , the cat said .",
        "Il gatto dell' anno è bello .",
        "word _ a b c",
    ] * 20


@pytest.fixture(scope="session")
def bert_tokenizer(tmp_path_factory):
    """WordPiece tokenizer with BERT special tokens ([CLS] ... [SEP]), cased."""
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    pieces = ["sky", "##scr", "##apers", "dell", "'", "s", "##s", "N", "##ew"]
    vocab = {tok: i for i, tok in enumerate(specials + WORDS + pieces)}
    for i in range(10):
        vocab.setdefault(f"w{i}", len(vocab))
    tok = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=False, strip_accents=False)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]",
    )


@pytest.fixture(scope="session")
def gpt2_tokenizer():
    """Byte-level BPE tokenizer like GPT-2 (no special tokens added, <|endoftext|> as BOS/EOS/PAD)."""
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2TokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator(_corpus(), trainer=trainer)
    fast = GPT2TokenizerFast(
        tokenizer_object=tok, bos_token="<|endoftext|>", eos_token="<|endoftext|>",
        unk_token="<|endoftext|>", add_prefix_space=True,
    )
    fast.pad_token = fast.eos_token  # what lib.data.load_model does
    return fast
