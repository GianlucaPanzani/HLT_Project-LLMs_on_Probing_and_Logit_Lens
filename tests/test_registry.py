from collections.abc import Iterator
from importlib import import_module
from pathlib import Path

import pytest

from lib import registry

# These values come from CONTRACTS.md, not from the YAML being tested
MODEL_CASES = [
    ("bert", "bert-base-cased", "encoder", "en", 12, 768, "float32"),
    ("distilbert", "distilbert-base-cased", "encoder", "en", 6, 768, "float32"),
    ("gpt2", "gpt2", "decoder", "en", 12, 768, "float32"),
    ("bert_it", "dbmdz/bert-base-italian-cased", "encoder", "it", 12, 768, "float32"),
    ("geppetto", "LorenzoDeMattei/GePpeTto", "decoder", "it", 12, 768, "float32"),
    ("mbert", "bert-base-multilingual-cased", "encoder", "multi", 12, 768, "float32"),
    ("mgpt", "ai-forever/mGPT", "decoder", "multi", 24, 2048, "float16"),
]
MODEL_KEYS = [case[0] for case in MODEL_CASES]

TREEBANK_CASES = [
    ("ewt", "UniversalDependencies/UD_English-EWT", "en_ewt-ud-{split}.conllu",
     "en", ["train", "dev", "test"], "ewt"),
    ("isdt", "UniversalDependencies/UD_Italian-ISDT", "it_isdt-ud-{split}.conllu",
     "it", ["train", "dev", "test"], "isdt"),
    ("pud_en", "UniversalDependencies/UD_English-PUD", "en_pud-ud-test.conllu",
     "en", ["test"], "ewt"),
    ("pud_it", "UniversalDependencies/UD_Italian-PUD", "it_pud-ud-test.conllu",
     "it", ["test"], "isdt"),
]
TREEBANK_KEYS = [case[0] for case in TREEBANK_CASES]


@pytest.fixture(autouse=True)
def clear_registry_cache() -> Iterator[None]:
    '''
    Clear cached YAML data before and after every test to isolate tests.

    Returns
    -------
    lifecycle (Iterator[None]): Yields control while pytest runs the test.
    '''
    # autouse=True applies this fixture even when a test does not name it.
    registry._get_yaml_file_content.cache_clear()
    try:
        yield
    finally:
        registry._get_yaml_file_content.cache_clear()


@pytest.fixture
def temporary_configs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    '''
    Redirect the registry to temporary configuration files for error tests.

    Parameters
    ----------
    - tmp_path (Path): Temporary directory created by pytest for this test.
    - monkeypatch (pytest.MonkeyPatch): Fixture that restores patched values
    automatically after the test.

    Returns
    -------
    configs_dir (Path): Temporary configs directory, separate from project files.
    '''
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    # Reproduce the src/lib/registry.py layout used to locate configs/.
    monkeypatch.setattr(registry, "__file__", str(tmp_path / "src" / "lib" / "registry.py"))
    return configs_dir


def test_list_models() -> None:
    '''
    Check that the registry lists exactly the seven model keys as strings.

    Returns
    -------
    None: Assertions fail if the returned type or model inventory is incorrect.
    '''
    keys = registry.list_models()
    assert isinstance(keys, list)
    assert all(isinstance(key, str) for key in keys)
    assert len(keys) == len(MODEL_KEYS)
    assert set(keys) == set(MODEL_KEYS)


# pytest runs this function once per row, reporting the model key on failure.
@pytest.mark.parametrize(
    "key,hf_id,kind,lang,num_layers,hidden_size,dtype", MODEL_CASES, ids=MODEL_KEYS
)
def test_model_specs(
    key: str, hf_id: str, kind: str, lang: str,
    num_layers: int, hidden_size: int, dtype: str,
) -> None:
    '''
    Check every model specification against the shared contract.

    Parameters
    ----------
    - key (str): Model registry key.
    - hf_id (str): Expected Hugging Face checkpoint identifier.
    - kind (str): Expected encoder or decoder architecture.
    - lang (str): Expected language code.
    - num_layers (int): Expected Transformer block count, excluding embeddings.
    - hidden_size (int): Expected hidden-state dimension.
    - dtype (str): Expected parameter precision.

    Returns
    -------
    None: Assertions fail for missing fields or incorrect model values.
    '''
    spec = registry.get_model_spec(key=key)
    expected = {
        "key": key, "hf_id": hf_id, "kind": kind, "lang": lang,
        "num_layers": num_layers, "hidden_size": hidden_size,
        "dtype": dtype, "revision": "main",
    }
    for field, value in expected.items():
        assert spec[field] == value, f"Incorrect {field} for {key}"


def test_unknown_model() -> None:
    '''
    Check that an unknown model raises ValueError with valid alternatives.

    Returns
    -------
    None: Assertions fail if the exception or its suggested keys are incorrect.
    '''
    # raises fails the test if the expected exception is not raised.
    with pytest.raises(ValueError) as error:
        registry.get_model_spec("unknown_model")
    assert "unknown_model" in str(error.value)
    for key in MODEL_KEYS:
        assert key in str(error.value)


@pytest.mark.parametrize(
    "key,repo,file_pattern,lang,splits,train_treebank", TREEBANK_CASES, ids=TREEBANK_KEYS
)
def test_treebank_specs(
    key: str, repo: str, file_pattern: str, lang: str,
    splits: list[str], train_treebank: str,
) -> None:
    '''
    Check treebank fields, official splits and training-treebank associations.

    Parameters
    ----------
    - key (str): Treebank registry key.
    - repo (str): Expected Universal Dependencies repository.
    - file_pattern (str): Expected CoNLL-U filename pattern.
    - lang (str): Expected language code.
    - splits (list[str]): Expected official dataset splits.
    - train_treebank (str): Expected treebank used to train probes.

    Returns
    -------
    None: Assertions fail for missing fields or incorrect treebank values.
    '''
    spec = registry.get_treebank_spec(key=key)
    expected = {
        "key": key, "repo": repo, "file_pattern": file_pattern,
        "lang": lang, "splits": splits, "train_treebank": train_treebank,
    }
    for field, value in expected.items():
        assert spec[field] == value, f"Incorrect {field} for {key}"


@pytest.mark.parametrize("key", ["unknown_treebank", "ud_release", "wikitext"])
def test_invalid_treebank(key: str) -> None:
    '''
    Reject unknown keys and YAML entries that are not UD treebanks.

    Parameters
    ----------
    - key (str): Invalid treebank key to request.

    Returns
    -------
    None: Assertions fail if the entry is accepted or valid suggestions are absent.
    '''
    with pytest.raises(ValueError) as error:
        registry.get_treebank_spec(key)
    message = str(error.value)
    assert key in message
    for valid_key in TREEBANK_KEYS:
        assert valid_key in message
    if key == "unknown_treebank":
        assert "ud_release" not in message
        assert "wikitext" not in message


def test_returned_values_do_not_modify_cache() -> None:
    '''
    Check that editing returned lists and specifications leaves the cache intact.

    Parameters
    ----------
    None.

    Returns
    -------
    None: Assertions fail if caller changes leak into subsequent registry calls.
    '''
    registry.list_models().clear()
    assert set(registry.list_models()) == set(MODEL_KEYS)

    model = registry.get_model_spec("gpt2")
    model["num_layers"] = -1
    assert registry.get_model_spec("gpt2")["num_layers"] == 12

    treebank = registry.get_treebank_spec("ewt")
    treebank["lang"] = "changed"
    # A shallow dict copy alone would still share this nested list.
    treebank["splits"].clear()
    fresh = registry.get_treebank_spec("ewt")
    assert fresh["lang"] == "en"
    assert fresh["splits"] == ["train", "dev", "test"]


def test_registry_from_another_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    '''
    Check that both registries work outside the repository's working directory.

    Parameters
    ----------
    - tmp_path (Path): Temporary directory unrelated to the repository.
    - monkeypatch (pytest.MonkeyPatch): Fixture that restores the working directory.

    Returns
    -------
    None: Assertions fail if configuration paths depend on the working directory.
    '''
    monkeypatch.chdir(tmp_path)
    assert registry.get_model_spec("gpt2")["hf_id"] == "gpt2"
    assert registry.get_treebank_spec("ewt")["train_treebank"] == "ewt"


@pytest.mark.parametrize("content", ["", "- item\n", "42\n"])
def test_yaml_requires_mapping(temporary_configs: Path, content: str) -> None:
    '''
    Reject empty YAML files and top-level lists or scalar values.

    Parameters
    ----------
    - temporary_configs (Path): Isolated configuration directory.
    - content (str): YAML content that does not represent a dictionary.

    Returns
    -------
    None: The test fails unless invalid configuration content raises ValueError.
    '''
    (temporary_configs / "invalid.yaml").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        registry._get_yaml_file_content("invalid.yaml")


def test_missing_yaml(temporary_configs: Path) -> None:
    '''
    Check that an absent configuration file raises FileNotFoundError.

    Parameters
    ----------
    - temporary_configs (Path): Empty configuration directory for this test.

    Returns
    -------
    None: The test fails unless the missing file is reported explicitly.
    '''
    assert not (temporary_configs / "missing.yaml").exists()
    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        registry._get_yaml_file_content("missing.yaml")


def test_yaml_cache(temporary_configs: Path) -> None:
    '''
    Check that repeated reads use cached data until the cache is cleared.

    Parameters
    ----------
    - temporary_configs (Path): Isolated directory for a small synthetic YAML file.

    Returns
    -------
    None: Assertions fail if caching or explicit cache invalidation does not work.
    '''
    path = temporary_configs / "example.yaml"
    path.write_text("value: 1\n", encoding="utf-8")
    assert registry._get_yaml_file_content("example.yaml") == {"value": 1}

    path.write_text("value: 2\n", encoding="utf-8")
    assert registry._get_yaml_file_content("example.yaml") == {"value": 1}

    registry._get_yaml_file_content.cache_clear()
    assert registry._get_yaml_file_content("example.yaml") == {"value": 2}


def test_upos_tags() -> None:
    '''
    Check the immutable UPOS sequence and the exact class-index order.

    Parameters
    ----------
    None.

    Returns
    -------
    None: The test fails for incorrect tags, or is skipped without loader dependencies.
    '''
    # Keep optional imports inside this test: skipping at module level would
    # also skip every registry test in CI, where torch is not installed.
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("conllu")
    data = import_module("lib.data")
    expected = (
        "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
        "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
    )
    assert isinstance(data.UPOS_TAGS, tuple)
    assert len(data.UPOS_TAGS) == 17
    assert data.UPOS_TAGS == expected


@pytest.mark.slow
@pytest.mark.parametrize("key", MODEL_KEYS)
def test_model_dimensions_match_hugging_face(key: str) -> None:
    '''
    Compare registered dimensions with real Hugging Face model configurations.

    Parameters
    ----------
    - key (str): Model registry key whose dimensions must be verified.

    Returns
    -------
    None: Assertions fail for mismatched dimensions; missing transformers skips the test.
    '''
    # slow marks the network-dependent test so CI can exclude it with -m "not slow".
    # AutoConfig downloads configuration metadata, never the model weights.
    transformers = pytest.importorskip("transformers")
    spec = registry.get_model_spec(key)
    config = transformers.AutoConfig.from_pretrained(spec["hf_id"], revision=spec["revision"])
    assert config.num_hidden_layers == spec["num_layers"]
    assert config.hidden_size == spec["hidden_size"]
