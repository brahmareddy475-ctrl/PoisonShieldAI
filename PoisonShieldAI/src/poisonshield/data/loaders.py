"""Offline dataset loading. Never captures packets or contacts remote hosts."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATASET_NAMES = {"synthetic", "cicids2017", "toniot", "unswnb15"}


def synthetic_flows(n: int = 850, seed: int = 0) -> pd.DataFrame:
    """Synthetic IDS-like data ONLY for functional tests; no scientific claims."""
    if n < 120:
        raise ValueError("Synthetic demonstration requires at least 120 records")
    rng = np.random.default_rng(seed)
    cls = rng.choice(["normal", "dos", "recon"], n, p=[0.50, 0.32, 0.18])
    offset = np.where(cls == "normal", 0.0, np.where(cls == "dos", 1.2, -0.9))
    noise = rng.normal(size=(n, 12))
    data = {}
    for j in range(12):
        scale = 0.5 + j / 22
        data[f"flow_stat_{j:02d}"] = (noise[:, j] * scale + offset * (0.8 if j < 7 else 0.35) + 3).clip(0)
    data["proto"] = np.where(cls == "dos", "udp", np.where(cls == "recon", "icmp", "tcp"))
    data["service"] = rng.choice(["http", "dns", "ssh"], n)
    data["Label"] = cls
    # Include harmless identifier columns to verify leakage-safe dropping.
    data["Source IP"] = [f"10.0.{i//250}.{i%250}" for i in range(n)]
    data["Timestamp"] = [f"2025-01-01 00:{i//60:02d}:{i%60:02d}" for i in range(n)]
    return pd.DataFrame(data)


def csv_files(path: str | Path, dataset: str) -> list[Path]:
    p = Path(path).expanduser()
    if p.is_file():
        if p.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, received {p}")
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"Dataset directory/file does not exist: {p}")
    candidates = sorted(p.rglob("*.csv"))
    if dataset == "cicids2017":
        candidates = [f for f in candidates if not any(s in f.name.lower() for s in ["features", "metadata", "groundtruth"]) ]
    elif dataset == "toniot":
        # A TON_IoT release includes heterogeneous telemetry and operating system logs.
        # Restrict to network data (or accept one explicit CSV file above).
        candidates = [f for f in candidates if "network" in f.name.lower()
                      or any("network" in folder.lower() for folder in f.relative_to(p).parts[:-1])]
    elif dataset == "unswnb15":
        candidates = [f for f in candidates if not any(s in f.name.lower() for s in ["features", "events", "groundtruth", "_gt."]) ]
        # Choose the official ready-to-use partition pair OR four raw files, never both.
        official = [f for f in candidates if "training-set" in f.name.lower() or "testing-set" in f.name.lower()]
        if official:
            candidates = official
    if not candidates:
        raise FileNotFoundError(f"No appropriate CSV files in {p} for {dataset}. See README dataset layout.")
    return candidates


def load_raw(dataset: str, path: str | None, max_rows_per_file: int | None = None,
             synthetic_rows: int = 850, synthetic_seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    if dataset not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset: {dataset}; expected {sorted(DATASET_NAMES)}")
    if dataset == "synthetic":
        return synthetic_flows(synthetic_rows, synthetic_seed), ["builtin:synthetic"]
    if not path:
        raise ValueError("A --data-path to locally downloaded CSV files is required for real datasets")
    files = csv_files(path, dataset)
    frames: list[pd.DataFrame] = []
    for f in files:
        frame = pd.read_csv(f, low_memory=False, nrows=max_rows_per_file, on_bad_lines="skip")
        frame.columns = [str(c).strip() for c in frame.columns]
        frame = frame.loc[:, ~frame.columns.duplicated()].copy()
        if frame.empty:
            continue
        # TON network boolean flags may contain "T"/"F"/"-", not floats.
        # Preserve them as binary numerical variables without one-hot explosion.
        if dataset == "toniot":
            for field in ["dns_AA", "dns_RD", "dns_RA", "dns_rejected",
                          "ssl_resumed", "ssl_established", "weird_notice"]:
                column = next((c for c in frame.columns if c.lower() == field.lower()), None)
                if column is not None:
                    as_text = frame[column].astype(str).str.strip().str.lower()
                    if set(as_text.dropna().unique()).issubset({"t", "f", "true", "false", "0", "1", "-", "nan", "none"}):
                        frame[column] = as_text.map({"t": 1., "true": 1., "1": 1.,
                            "f": 0., "false": 0., "0": 0., "-": 0.,
                            "nan": 0., "none": 0.}).astype(float)
        frames.append(frame)
    if not frames:
        raise ValueError("No CSV records were successfully read")
    # Multiple versions of the same dataset may be placed together by accident.
    if len(frames) > 1:
        common = set(frames[0].columns)
        for f in frames[1:]:
            common &= set(f.columns)
        if len(common) < 10:
            raise ValueError("CSV schemas differ substantially; supply a single consistent dataset variant")
        # Avoid all-NaN columns due to differing versions.
        cols = [c for c in frames[0] if c in common]
        frames = [f[cols] for f in frames]
    return pd.concat(frames, ignore_index=True), [str(f) for f in files]


def identify_labels(frame: pd.DataFrame, dataset: str, task: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if task not in {"binary", "multiclass"}:
        raise ValueError("task must be 'binary' or 'multiclass'")
    lowered = {str(c).strip().lower(): c for c in frame.columns}
    prefer = ("type", "attack_cat", "label") if dataset in {"toniot", "unswnb15"} else ("label", "attack_cat", "type")
    label_col = next((lowered[c] for c in prefer if c in lowered), None)
    if label_col is None:
        raise ValueError(f"No label column identified for {dataset}; found {list(frame.columns)[-12:]}")
    # When binary 0/1 and a categorical target coexist, use category for multiclass.
    if task == "multiclass":
        cat_col = next((lowered[c] for c in ("type", "attack_cat") if c in lowered), None)
        if cat_col is not None:
            label_col = cat_col
    raw_labels = frame[label_col].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    normal_names = {"normal", "benign", "0", "normal traffic", "non-attack", "nonattack"}
    if task == "binary":
        normal = raw_labels.isin(normal_names)
        y = np.where(normal, 0, 1).astype(np.int64)
        labels = ["normal", "attack"]
    else:
        if raw_labels.isin({"0", "1"}).all():
            raise ValueError("Multiclass task requires a class/category column, not only a binary 0/1 label")
        canonical = raw_labels.replace({"benign": "normal", "0": "normal", "normal traffic": "normal", "backdoors": "backdoor"})
        others = sorted(set(canonical.unique()) - {"normal"})
        if not others:
            raise ValueError("Cannot build multiclass task without attacks")
        labels = ["normal"] + others
        mapping = {name: idx for idx, name in enumerate(labels)}
        y = canonical.map(mapping).to_numpy(np.int64)
    if len(set(y)) < 2:
        raise ValueError("Dataset contains fewer than two label classes after filtering")
    # Remove ALL label aliases, including secondary binary label in a multiclass setup.
    forbidden = {"label", "type", "attack_cat", "attack_category", "category", "class", "class_label", "target", "binary_label"}
    features = frame.drop(columns=[c for c in frame.columns if str(c).strip().lower() in forbidden]).copy()
    return features, y, labels
