"""Italian alignment checks (issue X1): multiword tokens, elisions and clitics.

The offline test builds a tiny CoNLL-U file and checks how multiword tokens end up in the word table.
The slow tests download the four tokenizers used for Italian text and check that every UD word of
20 elision-rich sentences gets at least one subword, so no row of the feature files can be NaN
or shifted because of Italian morphology.

The sentences are real UD sentences (ISDT test, CC BY-NC-SA 4.0, and UD Italian-PUD, CC BY-SA 3.0),
written the way the word table stores them: multiword tokens are already split into their parts.
"""

import pytest

torch = pytest.importorskip("torch")

from lib.alignment import encode_words, pool_words  # noqa: E402
from lib.data import load_ud  # noqa: E402
from lib.registry import get_model_spec  # noqa: E402
from lib.word_table import build_word_table  # noqa: E402

ITALIAN_MODELS = ("bert_it", "geppetto", "mbert", "mgpt")

# sent_id -> UD words. Parts of multiword tokens: di+il (del), di+la (della), a+il (al), in+il (nel),
# di+l' (dell'), a+l' (all'), su+i (sui). Clitics are split into a truncated stem and the pronoun
# (riscattar+si, protegger+si, ricorda+ti, ricascati, ci).
SENTENCES = {
    "isst_tanl-1548": ["Ed", "ecco", "ci", "ricascati", "in", "il", '"', "politicamente", "corretto", '"', "."],
    "quest-1025": ["Quante", "persone", "morirono", "in", "l'", "incidente", "aereo", "di", "il", "1972", "in", "le", "Everglades", "?"],
    "quest-253": ["Che", "giorno", "è", "l'", "anniversario", "di", "la", "scoperta", "di", "l'", "America", "?"],
    "10_new-90": ["Il", "consiglio", "delibera", "a", "l'", "unanimità", "previa", "consultazione", "di", "il", "parlamento", "europeo", "."],
    "1_CC-36": ["Compensi", "per", "la", "comunicazione", "a", "il", "pubblico", "di", "l'", "opera", "mediante", "fonogrammi", "."],
    "isst_tanl-1147": ['"', "Chirac", ",", "ricorda", "ti", "Hiroshima", '"', "."],
    "isst_tanl-2083": ["È", "in", "la", "natura", "di", "le", "cose", "e", "di", "gli", "uomini", "."],
    "isst_tanl-2969": ["Da", "il", "1970", "è", "l'", "unico", "corpo", "militare", "di", "il", "Vaticano", "."],
    "isst_tanl-460": ["assassini", "e", "omicidi", "sono", "a", "l'", "ordine", "di", "il", "giorno", "tra", "i", "locali", ";"],
    "quest-787": ["Chi", "è", "il", "fondatore", "di", "l'", "Ordine", "di", "il", "Tempio", "Solare", "?"],
    "n01074011": ["Hanno", "un'", "unica", "possibilità", "per", "riscattar", "si", ":", "battere", "l'", "Inghilterra", "."],
    "n01003013": ["Forse", "l'", "abbigliamento", "richiesto", "era", "un", "po'", "a", "l'", "antica", "."],
    "n01128025": ["Innanzitutto", ",", "dobbiamo", "insegnare", "a", "le", "persone", "a", "protegger", "si", "meglio", "online", "."],
    "w01072046": ["Ma", "l'", "impatto", "di", "l'", "Hispania", "su", "i", "nuovi", "arrivati", "fu", "anche", "grande", "."],
    "n01092014": ["E", "che", "dire", "di", "la", "posizione", "di", "l'", "Australia", "?"],
    "w02004065": ["Ci", "sono", "differenti", "teorie", "circa", "le", "ragioni", "di", "l'", "abbandono", "di", "il", "luogo", "."],
    "n01002042": ["Le", "nuove", "spese", "sono", "alimentate", "da", "il", "considerevole", "conto", "bancario", "di", "la", "Clinton", "."],
    "n01070016": ["Ci", "sono", "più", "di", "330", "membri", "di", "l'", "equipaggio", "a", "bordo", "."],
    "n01089007": ["Naturalmente", ",", "in", "l'", "ufficio", "sono", "presenti", "molti", "vestiti", "e", "poco", "cibo", "."],
    "n01093025": ["Gli", "ambientalisti", "hanno", "accolto", "favorevolmente", "l'", "annuncio", "di", "la", "commissione", "."],
}

# Not in ISDT test or PUD IT: a curly apostrophe (6.6% of the apostrophes of ISDT) and a three-part clitic.
EXTRA_SENTENCES = {
    "curly_apostrophe": ["Il", "rumore", "di", "l’", "acqua", "."],
    "capitalised_part": ["In", "il", "mondo", "."],
    "three_part_clitic": ["Decise", "di", "andar", "se", "ne", "."],
}

CONLLU = """# sent_id = s1
# text = Il libro della nonna.
1\tIl\til\tDET\t_\t_\t2\tdet\t_\t_
2\tlibro\tlibro\tNOUN\t_\t_\t0\troot\t_\t_
3-4\tdella\t_\t_\t_\t_\t_\t_\t_\t_
3\tdi\tdi\tADP\t_\t_\t5\tcase\t_\t_
4\tla\tla\tDET\t_\t_\t5\tdet\t_\t_
5\tnonna\tnonna\tNOUN\t_\t_\t2\tnmod\t_\t_
6\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

# sent_id = s2
# text = Ha detto sì.
1\tHa\tavere\tAUX\t_\t_\t2\taux\t_\t_
2\tdetto\tdire\tVERB\t_\t_\t0\troot\t_\t_
2.1\tcosì\tcosì\tADV\t_\t_\t_\t_\t_\t_
3\tsì\tsì\tINTJ\t_\t_\t2\tdiscourse\t_\t_

"""


def test_multiword_tokens_are_not_rows_but_their_parts_are(tmp_path):
    (tmp_path / "it_test-ud-test.conllu").write_text(CONLLU, encoding="utf-8")
    sentences = load_ud(tmp_path, splits=("test",))["test"]
    table = build_word_table(sentences, "it")

    assert "della" not in set(table["form"])
    assert list(table["form"][:6]) == ["Il", "libro", "di", "la", "nonna", "."]
    assert "così" not in set(table["form"])  # empty node (id 2.1) is skipped
    assert list(table["upos"][2:4]) == ["ADP", "DET"]
    assert table.loc[2, "next_upos"] == "DET"  # first part of the multiword token: next word is its second part
    assert table["next_upos"].isna().sum() == 2  # one last word per sentence


@pytest.fixture(scope="module")
def tokenizers():
    transformers = pytest.importorskip("transformers")
    loaded = {}
    for key in ITALIAN_MODELS:
        spec = get_model_spec(key)
        decoder = spec["kind"] == "decoder"
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            spec["hf_id"], use_fast=True, **({"add_prefix_space": True} if decoder else {})
        )
        if tokenizer.pad_token_id is None:  # what lib.data.load_model does
            tokenizer.pad_token = tokenizer.eos_token
        loaded[key] = (tokenizer, decoder)
    return loaded


def _assert_one_vector_per_word(tokenizer, decoder, sentences):
    batch, word_ids = encode_words(tokenizer, sentences, add_bos=decoder, max_length=None)
    n_words = [len(s) for s in sentences]
    hidden = torch.arange(batch["input_ids"].shape[1], dtype=torch.float32).reshape(1, -1, 1).expand(len(sentences), -1, 2)
    for pooling in ("first", "last", "mean"):
        pooled = pool_words(hidden, word_ids, n_words, pooling)
        assert [p.shape[0] for p in pooled] == n_words
        for sentence, vectors in zip(sentences, pooled):
            assert not torch.isnan(vectors).any(), (pooling, sentence)
    for sentence, ids in zip(sentences, word_ids):
        seen = [w for w in ids if w is not None]
        assert sorted(set(seen)) == list(range(len(sentence))), sentence  # every word has a subword
        assert seen == sorted(seen), sentence  # words keep their order
    if decoder:
        assert all(ids[0] is None for ids in word_ids)  # BOS carries no word


@pytest.mark.slow
@pytest.mark.parametrize("key", ITALIAN_MODELS)
def test_elision_rich_sentences_get_one_vector_per_word(tokenizers, key):
    tokenizer, decoder = tokenizers[key]
    assert len(SENTENCES) == 20
    _assert_one_vector_per_word(tokenizer, decoder, list(SENTENCES.values()))


@pytest.mark.slow
@pytest.mark.parametrize("key", ITALIAN_MODELS)
def test_curly_apostrophe_capitalised_part_and_three_part_clitic(tokenizers, key):
    tokenizer, decoder = tokenizers[key]
    _assert_one_vector_per_word(tokenizer, decoder, list(EXTRA_SENTENCES.values()))


@pytest.mark.slow
def test_italian_decoders_use_prefix_space_for_every_word(tokenizers):
    for key in ("geppetto", "mgpt"):
        tokenizer, _ = tokenizers[key]
        assert tokenizer.add_prefix_space  # CONTRACTS.md section 1.1
        batch, word_ids = encode_words(tokenizer, [["Il", "gatto", "dell'", "anno"]], add_bos=True, max_length=None)
        pieces = tokenizer.convert_ids_to_tokens(batch["input_ids"][0].tolist())
        first_piece = {}
        for piece, word in zip(pieces, word_ids[0]):
            if word is not None:
                first_piece.setdefault(word, piece)
        assert len(first_piece) == 4 and all(p.startswith("Ġ") for p in first_piece.values()), pieces
