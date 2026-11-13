from functools import lru_cache
from pathlib import Path

import yaml


@lru_cache(maxsize=None)
def _get_yaml_file_content(filename: str) -> dict:
    '''
    Read and cache a YAML mapping from the repository's configs directory,
    locating the directory independently of the current working directory.

    Parameters
    ----------
    - filename (str): Configuration filename with the .yaml extension,
    such as "models.yaml" or "datasets.yaml".

    Returns
    -------
    content (dict): Parsed configuration mapping, cached by filename.
    '''
    if not filename.endswith(".yaml"):
        raise ValueError(f"The extension of the file has to be \".yaml\". File name provided: \"{filename}\".")
    
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    filepath = config_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"File \"{filename}\" not found.")
    
    with filepath.open(encoding="utf-8") as file:
        content = yaml.safe_load(file)

    if not isinstance(content, dict):
        raise ValueError(f"Expected a YAML mapping in {filepath}.")
    
    return content
    


def list_models() -> list[str]:
    '''
    List the model keys defined in the model registry.

    Returns
    -------
    keys (list[str]): Model keys in their YAML declaration order.
    '''
    models = _get_yaml_file_content("models.yaml")
    return list(models.keys())

def get_model_spec(key: str) -> dict:
    '''
    Retrieve a model specification by its registry key.
    Raise ValueError when the key is unknown.

    Parameters
    ----------
    - key (str): Model registry key, such as "bert" or "gpt2".

    Returns
    -------
    spec (dict): Model specification containing key, hf_id, kind, lang,
    num_layers, hidden_size, dtype and revision.
    '''
    models = _get_yaml_file_content("models.yaml")
    if key not in models:
        raise ValueError(f"Model \"{key}\" not found. Try with: {models.keys()}")
    
    return {
        "key": key,
        **models[key]
    }

def get_treebank_spec(key: str) -> dict:
    '''
    Retrieve a UD treebank specification by its registry key.
    Entries without train_treebank are excluded; an unknown key raises
    ValueError.

    Parameters
    ----------
    - key (str): Treebank registry key, such as "ewt" or "pud_it".

    Returns
    -------
    spec (dict): Treebank specification containing key, repo, file_pattern,
    lang, splits and train_treebank.
    '''
    datasets = _get_yaml_file_content("datasets.yaml")

    # Valid keys should be: ewt, isdt, pud_en, pud_it.
    valid_keys = [
        name for name, spec in datasets.items()
        if isinstance(spec, dict) and "train_treebank" in spec
    ]
    if key not in valid_keys:
        raise ValueError(f"Treebank \"{key}\" not found. Try with: {valid_keys}")
    
    return {
        "key": key,
        **datasets[key],
        "splits": list(datasets[key]["splits"]),  # override splits with a copy
    }
    
