"""Geração das Figuras 5.1, 5.2 e 5.3 da monografia a partir dos CSVs de resultados."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_FIGS = Path("figures")
OUT_FIGS.mkdir(parents=True, exist_ok=True)

def find_csv(name):
    for p in [Path("data_and_results") / name, Path("../data_and_results") / name, Path("tcc_validacao") / name, Path(name)]:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"Não foi possível localizar {name}")

def plot_all_figures():
    summary = find_csv("summary.csv")
    paired = find_csv("paired_intervals.csv")

    # Figura 5.1: Curvas Hit@k (k = 1, 5, 10, 20)
    ks = [1, 5, 10, 20]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    models = [("all-MiniLM-L6-v2", "MiniLM (all-MiniLM-L6-v2)"), ("BAAI/bge-small-en-v1.5", "BGE (BAAI/bge-small-en-v1.5)")]
    labels_map = {
        "baseline": ("Dense baseline", "-o", "#1f77b4"),
        "filters_full": ("Filters + full query", "--s", "#2ca02c"),
        "filters_soft": ("Filters + residual", ":^", "#d62728"),
    }
    for ax, (m_code, m_title) in zip(axes, models):
        sub = summary[summary["embedder"] == m_code]
        for sys_code, (label, style, color) in labels_map.items():
            row = sub[sub["system"] == sys_code]
            if not row.empty:
                vals = [float(row[f"Hit@{k}"].iloc[0]) * 100 for k in ks]
                ax.plot(ks, vals, style, color=color, label=label, linewidth=2, markersize=6)
        ax.set_title(m_title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Cutoff k", fontsize=10)
        ax.set_xticks(ks)
        ax.set_ylim(0, 65)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_ylabel("Single-target Hit@k (%)", fontsize=10)
    axes[1].legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_FIGS / "recreated_fig_5_1_recovery_curves.png", dpi=220)
    plt.close(fig)

    # Figura 5.2: Diferenças Pareadas e Intervalos de Confiança de 95% (Bootstrap)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, scale, unit in zip(axes, ["Hit@5", "MRR@20"], [100.0, 1.0], ["Delta Hit@5 (pp)", "Delta MRR@20"]):
        sub = paired[paired["metric"] == metric].copy()
        y_pos = np.arange(len(sub))
        deltas = sub["delta"].to_numpy() * scale
        ci_low = sub["ci_low"].to_numpy() * scale
        ci_high = sub["ci_high"].to_numpy() * scale
        err_low = deltas - ci_low
        err_high = ci_high - deltas
        labels = [f"{r.embedder.split('/')[-1]}\n({r.system})" for _, r in sub.iterrows()]
        ax.errorbar(deltas, y_pos, xerr=[err_low, err_high], fmt="o", color="#2b5c8f", ecolor="#2b5c8f", elinewidth=2, capsize=4)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(unit, fontsize=10)
        ax.set_title(f"Paired Differences vs. Baseline ({metric})", fontsize=10.5, fontweight="bold")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_FIGS / "recreated_fig_5_2_paired_intervals.png", dpi=220)
    plt.close(fig)
    print("Figuras recriadas com sucesso em figures/.")

if __name__ == "__main__":
    plot_all_figures()
