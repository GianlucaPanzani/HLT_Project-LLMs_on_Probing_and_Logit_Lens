# Italian alignment notes (issue X1)

Question: do the word table and the tokenizers handle the Italian cases that can break alignment (multiword tokens, elisions with apostrophes, clitics) before Italian features are extracted (E6)?
Data: UD r2.19, ISDT (train, dev, test) and Italian-PUD. Tokenizers: `bert_it`, `geppetto`, `mbert`, `mgpt` (Hugging Face ids in `configs/models.yaml`). English figures come from the same release (EWT, English-PUD).

## Summary

- **Alignment is safe.** For all four tokenizers, on all 322k words of ISDT and PUD IT, every UD word gets at least one subword: 0 zero-subword words, 0 truncated sentences, so no NaN rows are expected in E6. The longest sentence needs 403 positions (mBERT, ISDT train) against a limit of 512.
- **The word table is correct.** Multiword tokens are not rows; their parts are. The words of every table sentence equal the integer-id words of the CoNLL-U sentence (checked on all 14,167 ISDT and 1,000 PUD sentences). Empty nodes (37 in ISDT train, ids like `54.1`) are skipped.
- **Italian is much more affected than English by multiword splitting, and this affects the analyses, not the alignment.** 13.3% of Italian rows are parts of a multiword token (2.6% in EWT, 1.2% in English-PUD). 6.7% of Italian rows have a next word that belongs to the same multiword token, and their `next_upos` is `DET` in 94% of cases. This matters for the next-token task and for the lens bridge (section 5).
- **Two tokenizer quirks, no action needed in E1/E2:** mBERT maps the curly apostrophe `’` and the curly quotes `“ ”` to `[UNK]` (section 4), and the decoders always see a space before punctuation (section 5).

## 1. Multiword tokens

A multiword token (MWT) is a written word that UD splits into syntactic words, for example `della` = `di` + `la`. Only the parts are rows of the word table (CONTRACTS.md section 2).

| | ISDT (all splits) | PUD IT | EWT (all splits) | PUD EN |
| :-- | --: | --: | --: | --: |
| Sentences | 14,167 | 1,000 | | |
| Word rows | 298,338 | 23,783 | 254,820 | 21,180 |
| MWT | 19,888 (6.7% of rows) | 1,598 (6.7%) | 3,327 (1.3%) | 129 (0.6%) |
| Sentences with at least one MWT | 8,720 (61.6%) | 745 (74.5%) | | |
| Distinct MWT surface forms | 851 | 114 | | |
| Parts per MWT | 2: 19,862; 3: 26 | 2: 1,596; 3: 2 | | |
| Rows that are parts of an MWT | 39,802 (13.3%) | 3,198 (13.4%) | 6,656 (2.6%) | 258 (1.2%) |
| Rows whose next word is in the same MWT | 19,914 (6.7%) | 1,600 (6.7%) | | |
| `next_upos` of those rows | `DET` 94.4%, `PRON` 5.5% | `DET` 95.6%, `PRON` 4.4% | | |
| Empty nodes skipped | 37 | 0 | | |

The 20 most common MWTs in ISDT (the most common in PUD IT are almost the same forms):

| Surface | Count | Parts | Surface | Count | Parts |
| :-- | --: | :-- | :-- | --: | :-- |
| del | 2,972 | di il | all' | 505 | a l' |
| della | 2,305 | di la | nella | 462 | in la |
| dei | 1,335 | di i | degli | 449 | di gli |
| al | 1,287 | a il | dalla | 430 | da la |
| dell' | 1,064 | di l' | ai | 414 | a i |
| delle | 969 | di le | alle | 383 | a le |
| nel | 931 | in il | nei | 288 | in i |
| alla | 925 | a la | sul | 285 | su il |
| dal | 563 | da il | sulla | 251 | su la |
| | | | nell' | 245 | in l' |

Things to know about the parts:
- **The parts are not substrings of the surface form.** `del` becomes `di il`, `degli` becomes `di gli`, `dell'` becomes `di l'`.
- **Capitalisation goes to the first part:** `Nel` becomes `In il`.
- **Clitics become truncated stems.** `farlo` becomes `far lo`, `esserci` becomes `esser ci`, `muoversi` becomes `muover si`, and three-part clitics such as `andarsene` become `andar se ne` (26 cases in ISDT). The stems (`far`, `esser`, `muover`) are not normal Italian words. In ISDT, 18,796 MWTs (94.5%) are a preposition plus an article and 1,092 (5.5%) are clitic MWTs of this kind; in PUD IT the figures are 1,529 and 69 (4.3%).
- **Empty nodes are not MWTs.** They look like a tuple in the parser output, but with a `.` instead of a `-` (`54.1`). Code that tests only for a tuple id would count them as MWTs with no parts; `load_ud` handles them correctly.

## 2. Apostrophes and elisions

- 6,609 words of ISDT and 553 words of PUD IT end with an apostrophe. The most common in ISDT: `l'` 4,371, `L'` 394, `d'` 371, `un'` 329, `l’` 265, `c'` 145, `cos'` 56.
- The elided word is already its own UD word (`l'` and `amore` are two words). Together with `di l'` for `dell'`, an elision never has to be split by the tokenizer: it only has to tokenize `l'` into `l` + `'`, which all four tokenizers do.
- ISDT contains both the straight `'` (6,280 words) and the curly `’` (440 words, 6.6%). See section 4 for what mBERT does with `’`.

## 3. Tokenizer coverage

`encode_words` on every sentence of the word tables, with `add_bos=True` for the decoders and no truncation, to see how long the sequences get.

| Model | Words with no subword | Words with 2+ subwords (ISDT test) | Longest sequence, ISDT train | Limit |
| :-- | --: | --: | --: | --: |
| `bert_it` | 0 | 12.8% | 332 | 512 |
| `geppetto` | 0 | 12.4% | 327 | 1024 |
| `mbert` | 0 | 22.9% | 403 | 512 |
| `mgpt` | 0 | 25.2% | 400 | 2048 |

Zero words without a subword were found in ISDT train, dev, test and PUD IT, for any of the four tokenizers. So nothing is truncated and no row would be NaN. The concern in `docs/EXTRACTION.md` about the 310-word ISDT sentence is settled: it does not come near the limit with any tokenizer.

Subwords of five examples, as grouped by `encode_words` (a `+` joins the subwords of one word; `Ġ` marks a leading space in the byte-level BPE of the decoders):

| Case | `bert_it` | `geppetto` | `mbert` | `mgpt` |
| :-- | :-- | :-- | :-- | :-- |
| `di la`, `l'` in *E che dire di la posizione di l' Australia ?* | `di`, `la`, `l+'` | `Ġdi`, `Ġla`, `Ġl+'` | `di`, `la`, `l+'` | `Ġdi`, `Ġla`, `Ġl+'` |
| curly apostrophe: `l’` | `l+’` | `Ġl+âĢĻ` | `l+[UNK]` | `Ġl+âĢĻ` |
| capitalised part: `In il` | `In`, `il` | `ĠIn`, `Ġil` | `In`, `il` | `ĠIn`, `Ġil` |
| clitic: `far lo` | `far`, `lo` | `Ġfar`, `Ġlo` | `far`, `lo` | `Ġfar`, `Ġlo` |
| three parts: `andar se ne` | `andar`, `se`, `ne` | `Ġandar`, `Ġse`, `Ġne` | `anda+##r`, `se`, `ne` | `Ġand+ar`, `Ġse`, `Ġne` |

## 4. mBERT and the curly apostrophe

In ISDT train, mBERT produces 599 `[UNK]` positions: 436 come from words with `’` (`l’` 262, `d’` 55, `L’` 35, `quest’` 21, `c’` 19, ...) and 140 from the curly quotes `“` (71) and `”` (69). `bert_it` has 22 `[UNK]` positions (rare symbols such as `…`, `º`, Japanese characters), none from `’`. The two byte-level BPE tokenizers have no unknown tokens.

Consequences:
- Alignment is not affected: `l’` still has a subword (`l`) and the `[UNK]` is its second subword, so `first` pooling still returns the vector of `l`.
- With `mean` pooling the `[UNK]` is averaged in; with `last` pooling the word vector is the vector of `[UNK]`.
- A word that is only `“` or `”` becomes a single `[UNK]` input. These are 140 of 276,014 train rows (0.05%).
- No change is needed. If someone wants mBERT-specific handling, replacing `’` by `'` before tokenizing would remove this (it would change the input, not the word table).

## 5. Effect on the decoders and on next-token analysis

There are two separate effects. Both come from the contract (CONTRACTS.md section 3: pre-split UD words, `add_prefix_space=True`), and they apply to every model and language in the same way, except that multiword splitting is much more frequent in Italian.

**5.1 The model never sees the written text.**
- An Italian decoder reads `di la` where the text has `della`. This sequence does not occur in real Italian, because `di` + `la` is always written `della`. Encoders have the same issue.
- Natural vs. split tokenization on PUD IT test (ISDT test gives the same picture):
  - For `bert_it` and `mbert`, the tokenization of the UD words is identical to the tokenization of the original text in only 25% (PUD IT) and 35% (ISDT test) of the sentences, and it has about +1.5 to +1.6 tokens per sentence. When the MWTs are written in their surface form (`della` instead of `di la`), 99.1% (PUD IT) and 100.0% (ISDT test) of sentences are identical. So multiword splitting is the only difference for encoders, and `l'` + `amore` needs no repair.
  - For `geppetto` and `mgpt`, the tokenization is almost never identical to the natural text (0.0% to 0.1%, also with surface forms): `add_prefix_space=True` with pre-split words puts a space before every word, including punctuation (`Ġ.` instead of `.`). Punctuation is 11.3% of ISDT test rows and 9.7% of PUD IT rows.
  - The same prefix-space effect exists for `gpt2` in English.

**5.2 Next-token labels at multiword tokens.**
- For the 6.7% of rows whose next word is in the same MWT (for example `di` in `di la`), `next_upos` is the UPOS of the second part, `DET` in about 95% of cases.
- A decoder at the position of `di` only sees the left context. It cannot know that the original text was `della`, so it cannot predict the second part from the left. In text where `di` is a separate word, the next word is `DET` only about 25% of the time (ADP rows outside MWTs in ISDT: `NOUN` 35.6%, `DET` 25.2%, `PROPN` 13.6%, `VERB` 11.2%).
- Encoders see the right context and can simply read the next word.

So for Italian, `upos_next` accuracy of decoders is lowered by label noise that has nothing to do with syntax, encoders are favoured on the same rows, and English is affected much less (2.6% of EWT rows and 1.2% of English-PUD rows are MWT parts; English splits, such as contractions, were not examined in detail here). The same holds for the lens bridge (L8): at the position of a first part, the lens predicts the next word of the text-like context, while the label is the second part.

## 6. Recommendations

1. **E6 (feature extraction):** nothing to change. Run as planned; a `meta.json` with `n_nan_rows = 0` and `n_truncated_sentences = 0` is the expected outcome for all four Italian runs.
2. **Keep the contract (UD words as the unit),** and do not change `lib/alignment.py` for Italian.
3. **For P5/P4 and X2-X5 (`upos_next`):** report next-token results for Italian also on the rows that are not followed by a word of the same MWT (about 93% of the rows), next to the all-rows numbers. This needs a boolean column saying whether the next word is in the same MWT; it can be derived from the CoNLL-U `conllu_entries` already returned by `load_ud`. It is not in the word table today, and adding it would be a contract change, so it is a decision for the team.
4. **For L7/L8 (lens bridge):** same exclusion, and note in the results that the decoders read pre-split words with a space before every word (the contract), not the original text.
5. **When comparing English and Italian (X5):** say that 13.3% of Italian rows and 2.6% (EWT) or 1.2% (PUD EN) of English rows are parts of MWTs.
6. **Optional, not in the proposal:** run the lens (not the probes) on the original sentence text with offsets, as a robustness check of the next-token results for decoders.

## 7. How the numbers were obtained

Everything ran on the word tables and treebanks of UD r2.19 (`scripts/00_download_data.py`, `scripts/02_build_word_tables.py`). The tests that keep the conclusions of sections 1 and 3 checked are in `tests/test_italian_alignment.py`:

```bash
PYTHONPATH=src pytest tests/test_italian_alignment.py -q            # offline: MWT parts vs word table
PYTHONPATH=src pytest tests/test_italian_alignment.py -q -m slow    # downloads the 4 tokenizers
```

The counts of sections 1, 2, 4 and 5 were computed with throwaway scripts based on `lib.data.load_ud` (`conllu_entries`: an entry with id `(a, '-', b)` is an MWT, an entry with id `(a, '.', n)` is an empty node) and on `lib.alignment.encode_words`. For section 5.1 the natural tokenization is `tokenizer(text, add_special_tokens=False)` (with a leading space for the decoders), compared with `encode_words` without BOS, CLS and SEP, and the surface form of an MWT replaces its parts.
