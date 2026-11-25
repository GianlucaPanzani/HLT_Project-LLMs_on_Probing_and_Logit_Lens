"""Update Cells 8, 10, 12 in matteo_overlay_curves.ipynb to support use_tau=False (by layer)."""

import json
from pathlib import Path

nb_path = Path("notebooks/matteo_overlay_curves.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell_8_code = """def plot_model_overlay(model: str, eval_treebank: str, train_treebank: str, save_path: Path | None = None, use_tau: bool = False):
    data = load_overlay_data(model, eval_treebank, train_treebank)
    x_vals = data["taus"] if use_tau else data["layers"]
    xlabel = r"Relative Layer Depth $\\tau = l / L$" if use_tau else "Transformer Layer $l$"
    xlim = (-0.02, 1.02) if use_tau else (-0.4, 12.4)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    
    colors = {
        "probing": "#1f77b4",   # strong blue
        "logit": "#d62728",     # crimson red
        "tuned": "#2ca02c",     # forest green
        "jacobian": "#9467bd",  # purple
    }
    
    # -----------------------------------------------------
    # Panel (a): Accuracies and Probing Selectivity
    # -----------------------------------------------------
    # 1. Probing Selectivity
    ax1.plot(
        x_vals, data["probing_sel_next"], 
        label="Probing: UPOS Next Selectivity",
        color=colors["probing"], linewidth=2.5, marker="o", markersize=5
    )
    c_prob, _, _ = compute_commitment_layer(data["probing_sel_next"])
    c_prob_x = data["taus"][c_prob] if use_tau else data["layers"][c_prob]
    ax1.axvline(c_prob_x, color=colors["probing"], linestyle=":", alpha=0.6)
    
    # 2. Lenses UPOS Top-1 (dashed) and Word Top-1 (solid)
    for l_key, l_name in [("logit", "Logit Lens"), ("tuned", "Tuned Lens"), ("jacobian", "Jacobian Lens")]:
        # UPOS
        u_acc = data["lenses"][l_key]["upos_top1_acc"]
        ax1.plot(
            x_vals, u_acc,
            label=f"{l_name}: UPOS Acc",
            color=colors[l_key], linestyle="--", linewidth=2.0, marker="s", markersize=4
        )
        # Word Top-1
        w_acc = data["lenses"][l_key]["top1_acc"]
        ax1.plot(
            x_vals, w_acc,
            label=f"{l_name}: Token Top-1",
            color=colors[l_key], linestyle="-", linewidth=2.0, marker="^", markersize=4
        )
        
    ax1.set_title(f"{model.upper()} ({eval_treebank.upper()}) — Probing & Lens Commitments", fontsize=12, fontweight="bold")
    ax1.set_xlabel(xlabel, fontsize=11)
    ax1.set_ylabel("Accuracy / Selectivity", fontsize=11)
    ax1.set_xlim(*xlim)
    ax1.set_ylim(-0.02, 0.48)
    if not use_tau:
        ax1.set_xticks(range(0, 13))
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    
    # -----------------------------------------------------
    # Panel (b): Information Dynamics (Entropy & KL to Final)
    # -----------------------------------------------------
    ax2_twin = ax2.twinx()
    
    lines = []
    labels = []
    
    for l_key, l_name in [("logit", "Logit"), ("tuned", "Tuned"), ("jacobian", "Jacobian")]:
        ent = data["lenses"][l_key]["entropy_mean"]
        kl = data["lenses"][l_key]["kl_mean"]
        
        line_ent, = ax2.plot(x_vals, ent, color=colors[l_key], linestyle="-", linewidth=1.8, marker="o", markersize=4)
        line_kl, = ax2_twin.plot(x_vals, kl, color=colors[l_key], linestyle=":", linewidth=1.8, marker="x", markersize=4)
        
        lines.extend([line_ent, line_kl])
        labels.extend([f"{l_name} Entropy (H)", f"{l_name} KL to Final"])
        
    ax2.set_title(f"{model.upper()} ({eval_treebank.upper()}) — Entropy & KL Dynamics", fontsize=12, fontweight="bold")
    ax2.set_xlabel(xlabel, fontsize=11)
    ax2.set_ylabel("Shannon Entropy [nats] (Solid -)", fontsize=11)
    ax2_twin.set_ylabel(r"$D_{\\mathrm{KL}}(P_L \\parallel P_l)$ [nats] (Dotted :)", fontsize=11)
    ax2.set_xlim(*xlim)
    if not use_tau:
        ax2.set_xticks(range(0, 13))
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(lines, labels, loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Figure saved to: {save_path}")
        
    plt.show()
"""

cell_10_code = """# Overlay plot for GPT-2 by Transformer Layer l in [0, 12]
plot_model_overlay(
    "gpt2", "pud_en", "ewt",
    save_path=ROOT_DIR / "results" / "figures" / "ws4" / "gpt2_overlay.png",
    use_tau=False,
)
"""

cell_12_code = """# Overlay plot for GePpeTto by Transformer Layer l in [0, 12]
plot_model_overlay(
    "geppetto", "pud_it", "isdt",
    save_path=ROOT_DIR / "results" / "figures" / "ws4" / "geppetto_overlay.png",
    use_tau=False,
)
"""

nb["cells"][8]["source"] = [l + "\n" for l in cell_8_code.strip().split("\n")]
nb["cells"][10]["source"] = [l + "\n" for l in cell_10_code.strip().split("\n")]
nb["cells"][12]["source"] = [l + "\n" for l in cell_12_code.strip().split("\n")]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully updated Cells 8, 10, 12 in matteo_overlay_curves.ipynb!")
