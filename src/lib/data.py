from pathlib import Path

from conllu import parse_incr
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)


def load_ud(
    treebank_dir: str | Path,
    splits: tuple[str, ...] = ("train", "dev", "test"),
    max_sentences: int | None = None,
) -> dict[str, list[dict]]:
    '''
    Load annotated sentences from local Universal Dependencies CoNLL-U files,
    preserving the requested official dataset splits.

    Parameters
    ----------
    - treebank_dir (str | Path): Directory containing the treebank files.
    Exactly one file matching "*-ud-{split}.conllu" must exist for each
    requested split. The dataset must already be downloaded.
    - splits (tuple[str, ...]): Official splits to load. Each value must be
    "train", "dev" or "test". Defaults to ("train", "dev", "test").
    - max_sentences (int | None): Maximum number of nonempty annotated sentences
    to load per split. Must be positive when provided. If None, load all
    sentences. Defaults to None.

    Returns
    -------
    dict[str, list[dict]]: Dictionary mapping each requested split to a list
    of sentence records. Each record contains:
    - sent_id (str): Original sentence identifier, or a generated identifier
    based on the filename and sentence index when absent.
    - text (str | None): Original sentence text from the CoNLL-U metadata,
    or None when unavailable.
    - words (list[str]): Word forms from entries with integer IDs. Multiword
    token ranges and empty nodes are excluded from this list.
    - upos (list[str]): UPOS labels aligned with words. Missing word forms
    or UPOS annotations cause an error.
    - word_ids (list[int]): Original integer word IDs aligned with words.
    - conllu_entries (list[dict]): All parsed CoNLL-U entries, including
    multiword token ranges and empty nodes, retained for later alignment.
    '''
    treebank_dir = Path(treebank_dir)
    if not treebank_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {treebank_dir}")
    if max_sentences is not None and max_sentences < 1:
        raise ValueError("max_sentences must be positive.")

    dataset = {}
    for split in splits:
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        files = sorted(treebank_dir.glob(f"*-ud-{split}.conllu"))
        if len(files) != 1:
            raise ValueError(f"Expected one file for split '{split}' in {treebank_dir}, found {len(files)}.")

        sentences = []
        with files[0].open(encoding="utf-8") as file:
            for sentence_index, sentence in enumerate(parse_incr(file)):
                words = [token for token in sentence if isinstance(token["id"], int)]
                if not words:
                    continue

                sent_id = sentence.metadata.get("sent_id", f"{files[0].stem}:{sentence_index}")

                if any(word["upos"] in (None, "_") or word["form"] in (None, "_") for word in words):
                    raise ValueError(f"Missing word or UPOS annotation in {sent_id}.")

                sentences.append({
                    "sent_id": sent_id,
                    "text": sentence.metadata.get("text"),
                    "words": [word["form"] for word in words],
                    "upos": [word["upos"] for word in words],
                    "word_ids": [word["id"] for word in words],
                    "conllu_entries": list(sentence),
                })

                if (
                    max_sentences is not None
                    and len(sentences) >= max_sentences
                ):
                    break

        dataset[split] = sentences

    return dataset



def load_model(
    checkpoint: str,
    *,
    kind: str,
    random_weights: bool = False,
    seed: int = 42,
    device: str | None = None,
    revision: str = "main",
    tokenizer_kwargs: dict | None = None,
):
    '''
    Load a encoder (bidirectional LM) or decoder (causal LM), freeze all
    model parameters and enable evaluation mode.

    Parameters
    ----------
    - checkpoint (str): Hugging Face repository ID or local directory containing
    the model configuration and, when random_weights=False, pretrained weights.
    The same location is used to load the tokenizer.
    - kind (str): Model category; must be "encoder" or "causal". Causal models
    are loaded with their language modeling output head.
    - random_weights (bool): If True, initialize a new model from the checkpoint
    configuration instead of loading pretrained weights. The tokenizer is
    still loaded from the checkpoint. Defaults to False.
    - seed (int): Random seed used when random_weights=True. Ignored when
    loading pretrained weights. Defaults to 42.
    - device (str | None): Device on which to place the model, such as "cpu",
    "cuda" or "mps". If None, select CUDA when available, otherwise MPS,
    otherwise CPU. Defaults to None.
    - revision (str): Hugging Face repository branch, tag or commit hash to
    load. Applies to remote checkpoints. Defaults to "main".
    - tokenizer_kwargs (dict | None): Additional tokenizer loading options,
    such as {"add_prefix_space": True} for pretokenized GPT-2 inputs.
    A fast tokenizer is required. Defaults to None.

    Returns
    -------
    A tuple containing:
    - model (PreTrainedModel): Model on the selected device, with frozen
    parameters, evaluation mode enabled and hidden-state output enabled.
    Causal models include their output head and have caching disabled.
    - tokenizer (PreTrainedTokenizerFast): Tokenizer configured for right
    padding. If no padding token exists, the EOS token is reused.
    - metadata (dict): Checkpoint, revision, model category, random initialization
    flag, initialization seed or None, device, number of Transformer layers,
    hidden-state dimension and vocabulary size.
    '''
    model_classes = {
        "encoder": AutoModel,
        "decoder": AutoModelForCausalLM,
    }

    # Select the model
    if kind not in model_classes:
        raise ValueError("kind must be 'encoder' or 'decoder'.")
    model_class = model_classes[kind]

    # Choose the available device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Create the tokenizer that will be used
    tokenizer_options = {"use_fast": True}
    tokenizer_options.update(tokenizer_kwargs or {})
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        revision=revision,
        **tokenizer_options
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a PAD nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for word alignment.")

    # Create the configuration for the model
    config = AutoConfig.from_pretrained(
        checkpoint,
        revision=revision
    )
    config.output_hidden_states = True
    config.return_dict = True
    if kind == "decoder":
        config.use_cache = False
    config.pad_token_id = tokenizer.pad_token_id

    # Create the model
    model = model_class.from_pretrained(
        checkpoint,
        revision=revision,
        config=config,
    )
    if random_weights: # it randomize the weights of the pretrained model but for our purposes it should be always False
        with torch.random.fork_rng(): # preserve the caller's random-number generator state
            torch.manual_seed(seed)
            model = model_class.from_config(config)
    model.requires_grad_(False)
    model.eval()
    model.to(device)

    metadata = {
        "checkpoint": checkpoint,
        "revision": revision,
        "kind": kind,
        "random_weights": random_weights,
        "seed": seed if random_weights else None,
        "device": str(device),
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
    }

    return model, tokenizer, metadata