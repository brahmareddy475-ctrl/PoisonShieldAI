"""600-dpi publication-style plots, with no hardcoded experimental results."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_run(results: pd.DataFrame, output_dir: Path) -> None:
    if results.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for victim, frame in results[results.status == "ok"].groupby("victim"):
        methods = list(frame.method)
        values = frame.macro_f1.astype(float).values
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.bar(np.arange(len(methods)), values)
        ax.set_xticks(np.arange(len(methods)), methods, rotation=45, ha="right")
        ax.set_ylabel("Macro-F1 on held-out clean test data")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir/f"{victim}_method_macro_f1.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


def plot_sweep(results: pd.DataFrame, output_dir: Path) -> None:
    if results.empty or results.poison_rate.nunique() < 2:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for (dataset, victim), subset in results[results.status == "ok"].groupby(["dataset", "victim"]):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for method, frame in subset.groupby("method"):
            if method not in {"clean", "poisoned", "PoisonShieldAI", "DBSCAN_only", "Discriminator_only"}:
                continue
            grouped = frame.groupby("poison_rate").macro_f1
            means = grouped.mean().sort_index()
            stds = grouped.std().reindex(means.index).fillna(0.)
            ax.errorbar(100*means.index.to_numpy(), means.to_numpy(), yerr=stds.to_numpy(),
                        marker="o", capsize=3, label=method)
        ax.set_xlabel("Poisoning budget (% of clean victim training records)")
        ax.set_ylabel("Macro-F1 on held-out clean test data")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(output_dir/f"{dataset}_{victim}_rate_sweep.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
