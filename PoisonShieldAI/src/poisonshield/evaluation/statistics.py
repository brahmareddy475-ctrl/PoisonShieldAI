"""Five-seed aggregate statistics and optional paired Holm-adjusted Wilcoxon tests."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon, rankdata


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    group = ["dataset", "task", "victim", "poison_rate", "method"]
    measure = ["accuracy", "macro_f1", "weighted_f1", "clean_retention", "poison_detection_recall"]
    rows = []
    for keys, subset in results.groupby(group, dropna=False):
        successful = subset[subset["status"] == "ok"]
        for metric in measure:
            values = successful[metric].dropna().astype(float).values
            if len(values) == 0:
                continue
            n = len(values)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
            radius = float(t.ppf(0.975, n-1) * sd / np.sqrt(n)) if n > 1 else np.nan
            rows.append(dict(zip(group, keys)) | {
                "metric": metric, "n": n, "mean": mean, "sd": sd,
                "ci95_low": mean-radius if n > 1 else np.nan,
                "ci95_high": mean+radius if n > 1 else np.nan,
            })
    return pd.DataFrame(rows)


def paired_tests(results: pd.DataFrame, reference: str = "PoisonShieldAI",
                 metric: str = "macro_f1") -> pd.DataFrame:
    """Wilcoxon signed rank, paired by seed. Holm within each dataset/victim/budget family."""
    if results.empty:
        return pd.DataFrame()
    group = ["dataset", "task", "victim", "poison_rate"]
    rows = []
    for keys, subset in results[results.status == "ok"].groupby(group):
        reference_rows = subset[subset.method == reference][["seed",metric]].rename(columns={metric:"ref"})
        local = []
        for method in sorted(set(subset.method) - {reference, "clean"}):
            other = subset[subset.method == method][["seed",metric]].rename(columns={metric:"other"})
            joined = reference_rows.merge(other, on="seed").dropna()
            if len(joined) < 5:
                continue
            delta = joined["ref"].values - joined["other"].values
            try:
                p = float(wilcoxon(delta, zero_method="pratt", alternative="two-sided").pvalue)
            except ValueError:
                p = 1.0
            delta_sd = float(np.std(delta, ddof=1))
            dz = float(np.mean(delta) / delta_sd) if delta_sd > 0 else np.nan
            nonzero = delta[delta != 0]
            if len(nonzero):
                signed_ranks = rankdata(np.abs(nonzero))
                rank_biserial = float((signed_ranks[nonzero > 0].sum() -
                                       signed_ranks[nonzero < 0].sum()) / signed_ranks.sum())
            else:
                rank_biserial = 0.0
            local.append(dict(zip(group, keys)) | {"comparison": f"{reference} vs {method}",
                    "n_paired": len(delta), "mean_paired_delta": float(delta.mean()),
                    "cohens_dz": dz, "rank_biserial": rank_biserial,
                    "p_raw": p})
        ranked = sorted(range(len(local)), key=lambda i: local[i]["p_raw"])
        # Holm step-down adjustment is monotone in ascending raw-p order.
        running = 0.
        for rank, idx in enumerate(ranked):
            running = max(running, min(1., (len(local)-rank)*local[idx]["p_raw"]))
            local[idx]["p_holm"] = running
        rows.extend(local)
    return pd.DataFrame(rows)
