"""Layer mapping comparison between BERT and DistilBERT.

Builds tables comparing the rigid 1:2 layer mapping with the CKA-best
mapping in terms of UPOS accuracy, selectivity and CoG_norm (C4).
"""

from __future__ import annotations

import pandas as pd

from lib.metrics import center_of_gravity


def _extract_layer_metrics(
    probing_df: pd.DataFrame,
    model: str,
    treebank: str = "ewt",
    probe: str = "linear",
    init: str = "pretrained",
    pooling: str = "first",
) -> pd.DataFrame:
    """Extract per-layer accuracy and selectivity from a probing results frame.

    Parameters
    ----------
    probing_df : pd.DataFrame
        Probing results following CONTRACTS.md §6.
    model : str
        Model registry key to filter on.
    treebank : str
        Eval treebank to filter on.
    probe : str
        Probe type to filter on.
    init : str
        Initialisation to filter on.
    pooling : str
        Pooling strategy to filter on.

    Returns
    -------
    pd.DataFrame
        Columns: ``layer``, ``acc_upos``, ``acc_control``, ``selectivity``,
        ``tau``, sorted by layer.
    """
    mask = (
        (probing_df["model"] == model)
        & (probing_df["eval_treebank"] == treebank)
        & (probing_df["probe"] == probe)
        & (probing_df["init"] == init)
        & (probing_df["pooling"] == pooling)
    )
    df = probing_df.loc[mask].copy()

    if df.empty:
        raise ValueError(
            f"No probing results for model={model!r}, eval_treebank={treebank!r}, "
            f"probe={probe!r}, init={init!r}, pooling={pooling!r}."
        )

    upos = df.loc[df["task"] == "upos", ["layer", "acc"]].groupby("layer", sort=True)["acc"].mean()
    ctrl = df.loc[df["task"] == "control", ["layer", "acc"]].groupby("layer", sort=True)["acc"].mean()

    if upos.empty or ctrl.empty:
        raise ValueError(f"Missing upos or control task rows for model={model!r}.")

    merged = pd.DataFrame({"acc_upos": upos, "acc_control": ctrl}).sort_index()
    merged["selectivity"] = merged["acc_upos"] - merged["acc_control"]
    num_layers = int(merged.index.max())
    merged["tau"] = merged.index / num_layers
    merged = merged.reset_index().rename(columns={"index": "layer"})
    return merged


def build_layer_mapping_table(
    probing_bert: pd.DataFrame,
    probing_distilbert: pd.DataFrame,
    cka_df: pd.DataFrame,
    *,
    treebank: str = "ewt",
    probe: str = "linear",
    init: str = "pretrained",
    pooling: str = "first",
) -> pd.DataFrame:
    """Build the 1:2 and CKA-best layer mapping comparison table.

    For each DistilBERT layer ``l`` (0..6), shows the rigid BERT match
    ``2l``, the CKA-best BERT match, and their UPOS accuracy, selectivity
    and CKA values.

    Parameters
    ----------
    probing_bert : pd.DataFrame
        Probing results for BERT (CONTRACTS.md §6).
    probing_distilbert : pd.DataFrame
        Probing results for DistilBERT (CONTRACTS.md §6).
    cka_df : pd.DataFrame
        CKA matrix (CONTRACTS.md §8).
    treebank : str
        Eval treebank to filter on.
    probe : str
        Probe type.
    init : str
        Initialisation variant.
    pooling : str
        Pooling strategy.

    Returns
    -------
    pd.DataFrame
        One row per DistilBERT layer with columns describing both
        mappings, accuracy, selectivity and CKA values.
    """
    bert_metrics = _extract_layer_metrics(
        probing_bert, "bert", treebank=treebank, probe=probe, init=init, pooling=pooling,
    )
    distil_metrics = _extract_layer_metrics(
        probing_distilbert, "distilbert", treebank=treebank, probe=probe, init=init, pooling=pooling,
    )

    records: list[dict] = []
    for _, row_d in distil_metrics.iterrows():
        lb = int(row_d["layer"])
        rigid_la = 2 * lb  # 1:2 mapping

        # Rigid mapping BERT metrics
        bert_rigid = bert_metrics.loc[bert_metrics["layer"] == rigid_la]
        if bert_rigid.empty:
            raise ValueError(f"BERT layer {rigid_la} not found in probing results.")
        bert_rigid = bert_rigid.iloc[0]

        # CKA-best mapping
        cka_col = cka_df.loc[cka_df["layer_b"] == lb]
        if cka_col.empty:
            raise ValueError(f"DistilBERT layer {lb} not found in CKA matrix.")
        best_row = cka_col.loc[cka_col["cka"].idxmax()]
        best_la = int(best_row["layer_a"])
        best_cka_val = float(best_row["cka"])

        bert_best = bert_metrics.loc[bert_metrics["layer"] == best_la]
        if bert_best.empty:
            raise ValueError(f"BERT layer {best_la} not found in probing results.")
        bert_best = bert_best.iloc[0]

        records.append({
            "distilbert_layer": lb,
            "distilbert_acc": float(row_d["acc_upos"]),
            "distilbert_sel": float(row_d["selectivity"]),
            # Rigid 1:2 mapping
            "rigid_bert_layer": rigid_la,
            "rigid_bert_acc": float(bert_rigid["acc_upos"]),
            "rigid_bert_sel": float(bert_rigid["selectivity"]),
            "rigid_acc_diff": float(row_d["acc_upos"] - bert_rigid["acc_upos"]),
            "rigid_sel_diff": float(row_d["selectivity"] - bert_rigid["selectivity"]),
            # CKA-best mapping
            "cka_best_bert_layer": best_la,
            "cka_best_bert_acc": float(bert_best["acc_upos"]),
            "cka_best_bert_sel": float(bert_best["selectivity"]),
            "cka_best_cka": best_cka_val,
            "cka_best_acc_diff": float(row_d["acc_upos"] - bert_best["acc_upos"]),
            "cka_best_sel_diff": float(row_d["selectivity"] - bert_best["selectivity"]),
        })

    return pd.DataFrame(records)


def compute_cog_comparison(
    probing_bert: pd.DataFrame,
    probing_distilbert: pd.DataFrame,
    *,
    treebank: str = "ewt",
    probe: str = "linear",
    init: str = "pretrained",
    pooling: str = "first",
) -> dict:
    """Compute CoG_norm for BERT and DistilBERT selectivity curves.

    Parameters
    ----------
    probing_bert : pd.DataFrame
        Probing results for BERT.
    probing_distilbert : pd.DataFrame
        Probing results for DistilBERT.
    treebank : str
        Eval treebank.
    probe : str
        Probe type.
    init : str
        Initialisation variant.
    pooling : str
        Pooling strategy.

    Returns
    -------
    dict
        Keys: ``bert_cog_norm``, ``distilbert_cog_norm``,
        ``bert_selectivity`` (array), ``distilbert_selectivity`` (array),
        ``bert_tau`` (array), ``distilbert_tau`` (array).
    """
    bert_m = _extract_layer_metrics(
        probing_bert, "bert", treebank=treebank, probe=probe, init=init, pooling=pooling,
    )
    distil_m = _extract_layer_metrics(
        probing_distilbert, "distilbert", treebank=treebank, probe=probe, init=init, pooling=pooling,
    )

    bert_sel = bert_m["selectivity"].to_numpy()
    distil_sel = distil_m["selectivity"].to_numpy()

    return {
        "bert_cog_norm": center_of_gravity(bert_sel, normalize=True),
        "distilbert_cog_norm": center_of_gravity(distil_sel, normalize=True),
        "bert_selectivity": bert_sel,
        "distilbert_selectivity": distil_sel,
        "bert_tau": bert_m["tau"].to_numpy(),
        "distilbert_tau": distil_m["tau"].to_numpy(),
    }
