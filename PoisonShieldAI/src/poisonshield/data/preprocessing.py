"""Train-only fitted feature engineering with reversible schema metadata."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


ID_FIELDS = {
    "flow id", "source ip", "destination ip", "source port", "destination port",
    "src_ip", "dst_ip", "src_port", "dst_port", "sport", "dsport",
    "timestamp", "time", "ts", "stime", "ltime", "id", "srcip", "dstip",
    "flow_id", "sourceip", "destinationip"
}
CATEGORICAL_FIELDS = {
    "protocol", "proto", "service", "conn_state", "state", "ssl_version",
    "ssl_cipher", "http_method", "http_version", "http_orig_mime_types",
    "http_resp_mime_types", "weird_name", "weird_addl"
}
DROP_TEXT_FIELDS = {
    "dns_query", "ssl_subject", "ssl_issuer", "http_uri", "http_user_agent",
    "attack_cat", "label", "type"
}


@dataclass
class Partition:
    X: np.ndarray
    y: np.ndarray
    ids: np.ndarray


@dataclass
class PreparedData:
    attacker: Partition
    victim: Partition
    trusted: Partition
    test: Partition
    schema: "FeaturePreprocessor"
    classes: list[str]
    source_names: list[str]
    total_clean_records: int


class FeaturePreprocessor:
    def __init__(self, rare_fraction: float = 0.001, high_cardinality: int = 150):
        self.rare_fraction = float(rare_fraction)
        self.high_cardinality = int(high_cardinality)
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.categories: dict[str, list[str]] = {}
        self.scaler = MinMaxScaler(clip=True)
        self.columns: list[str] = []
        self.numeric_indices: list[int] = []
        self.categorical_groups: list[list[int]] = []
        self.dropped_columns: dict[str, str] = {}
        self.fitted = False

    @property
    def n_features(self) -> int:
        return len(self.columns)

    @staticmethod
    def _cat_strings(values: pd.Series) -> pd.Series:
        return values.fillna("__missing__").astype(str).str.strip().str.lower().replace("", "__missing__")

    def fit(self, x: pd.DataFrame) -> "FeaturePreprocessor":
        self.dropped_columns = {}
        self.numeric_cols = []
        self.categorical_cols = []
        self.categories = {}
        self.numeric_indices = []
        self.categorical_groups = []
        self.columns = []
        for col in x.columns:
            key = col.strip().lower()
            if key in ID_FIELDS or key in DROP_TEXT_FIELDS or key.endswith(".1"):
                self.dropped_columns[col] = "identifier, free text, or duplicate-name suffix"
                continue
            if x[col].nunique(dropna=False) <= 1:
                self.dropped_columns[col] = "constant"
                continue
            series = x[col]
            numeric_candidate = (key not in CATEGORICAL_FIELDS) and pd.api.types.is_numeric_dtype(series)
            if not numeric_candidate and key not in CATEGORICAL_FIELDS:
                coerced = pd.to_numeric(series, errors="coerce")
                numeric_candidate = float(coerced.notna().mean()) > 0.995
            if numeric_candidate:
                self.numeric_cols.append(col)
            else:
                if series.nunique(dropna=True) > self.high_cardinality:
                    self.dropped_columns[col] = "high-cardinality text"
                else:
                    self.categorical_cols.append(col)
        if not (self.numeric_cols or self.categorical_cols):
            raise ValueError("Preprocessing removed all candidate features")
        # Duplicate *value* columns: first detect cheap sample signatures, then verify full equality.
        candidates = self.numeric_cols + self.categorical_cols
        dup_cols = set()
        fingerprints: dict[tuple, str] = {}
        sample = x.head(min(len(x), 320))
        for c in candidates:
            signature = tuple(sample[c].astype(str).tolist())
            if signature in fingerprints and x[c].equals(x[fingerprints[signature]]):
                dup_cols.add(c)
                self.dropped_columns[c] = f"duplicate of {fingerprints[signature]}"
            else:
                fingerprints[signature] = c
        self.numeric_cols = [c for c in self.numeric_cols if c not in dup_cols]
        self.categorical_cols = [c for c in self.categorical_cols if c not in dup_cols]
        for c in self.numeric_cols:
            self.numeric_indices.append(len(self.columns))
            self.columns.append(c)
        if self.numeric_cols:
            numeric = x[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
            if not np.isfinite(numeric.to_numpy(dtype=float)).all():
                raise ValueError("Training data has missing or infinite numerical values")
            self.scaler.fit(numeric.to_numpy(dtype=np.float64))
        for c in self.categorical_cols:
            vals = self._cat_strings(x[c])
            counts = vals.value_counts()
            threshold = self.rare_fraction * len(vals)
            valid = sorted([str(v) for v, n in counts.items() if n >= threshold])
            rare = [str(v) for v, n in counts.items() if n < threshold]
            if not valid:
                # Extremely small datasets: retaining all categories is safer.
                valid = sorted(counts.index.astype(str).tolist())
                rare = []
            if rare:
                valid.append("__other__")
            self.categories[c] = valid
            positions = list(range(len(self.columns), len(self.columns) + len(valid)))
            self.categorical_groups.append(positions)
            self.columns.extend(f"{c}={v}" for v in valid)
        self.fitted = True
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() before transform()")
        missing = [c for c in self.numeric_cols + self.categorical_cols if c not in x.columns]
        if missing:
            raise ValueError(f"Missing schema features: {missing}")
        parts = []
        if self.numeric_cols:
            numeric = x[self.numeric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
            if not np.isfinite(numeric).all():
                raise ValueError("Missing/infinite numerical values in input features")
            parts.append(np.clip(self.scaler.transform(numeric), 0.0, 1.0).astype(np.float32))
        for c in self.categorical_cols:
            vals = self._cat_strings(x[c])
            cats = self.categories[c]
            loc = {name: i for i, name in enumerate(cats)}
            fallback = loc.get("__other__", -1)
            indices = vals.map(lambda v: loc.get(v, fallback)).to_numpy(dtype=np.int64)
            one_hot = np.zeros((len(x), len(cats)), dtype=np.float32)
            is_known = indices >= 0
            # Unseen values become all-zero only when no rare-category other
            # bucket was fitted (as with OneHotEncoder(handle_unknown='ignore')).
            one_hot[np.arange(len(x))[is_known], indices[is_known]] = 1.0
            parts.append(one_hot)
        return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)

    def metadata(self) -> dict:
        if not self.fitted:
            raise RuntimeError("Unfitted feature preprocessor")
        return {
            "feature_names": self.columns, "numeric_features": self.numeric_cols,
            "categorical_features": self.categorical_cols, "onehot_categories": self.categories,
            "numeric_indices": self.numeric_indices, "categorical_groups": self.categorical_groups,
            "dropped_features": self.dropped_columns,
            "numeric_min": self.scaler.data_min_.tolist() if self.numeric_cols else [],
            "numeric_max": self.scaler.data_max_.tolist() if self.numeric_cols else [],
            "scaling_fit_on": "clean 80% training split only",
        }


def _split(idx: np.ndarray, labels: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    subset_labels = labels[idx]
    counts = np.bincount(subset_labels)
    counts = counts[counts > 0]
    n_holdout = int(np.ceil(test_size * len(idx)))
    num_classes = len(counts)
    strat = subset_labels if len(counts) >= 2 and counts.min() >= 2 and n_holdout >= num_classes and (len(idx)-n_holdout)>=num_classes else None
    return train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)


def prepare_dataset(raw: pd.DataFrame, dataset: str, task: str,
                    split_seed: int = 0, source_names: list[str] | None = None) -> PreparedData:
    from .loaders import identify_labels
    # Drop full duplicate rows BEFORE partitioning to avoid exact train/test duplicates.
    original = len(raw)
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").drop_duplicates().reset_index(drop=True)
    if len(raw) < 100:
        raise ValueError(f"Only {len(raw)} clean rows remain after removing missing/duplicate records (from {original})")
    features, y, names = identify_labels(raw, dataset, task)
    all_idx = np.arange(len(raw))
    train_80, tmp_20 = _split(all_idx, y, test_size=0.2, seed=split_seed)
    trust_idx, test_idx = _split(tmp_20, y, test_size=0.5, seed=split_seed)
    attacker_idx, victim_idx = _split(train_80, y, test_size=0.5, seed=split_seed)
    schema = FeaturePreprocessor().fit(features.iloc[train_80])
    def part(i: np.ndarray) -> Partition:
        return Partition(schema.transform(features.iloc[i]), y[i].astype(np.int64),
                         np.asarray([f"clean:{j}" for j in i], dtype=str))
    return PreparedData(part(attacker_idx), part(victim_idx), part(trust_idx),
                        part(test_idx), schema, names, source_names or [], len(raw))


def save_manifest(prepared: PreparedData, path: str | Path, dataset: str | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    expected = {"cicids2017": 71, "toniot": 76}.get(dataset)
    record = {
        "dataset": dataset,
        "feature_count": prepared.schema.n_features,
        "manuscript_feature_count_reference": expected,
        "matches_manuscript_feature_count": prepared.schema.n_features == expected if expected is not None else None,
        "note_on_feature_count": "Feature counts depend on the exact public CSV release, missing/constant columns, and train-only fitted vocabularies; counts are never forcibly adjusted.",
        "total_clean_records": prepared.total_clean_records,
        "source_files": prepared.source_names, "classes": prepared.classes,
        "attacker_rows": len(prepared.attacker.y),
        "victim_rows": len(prepared.victim.y),
        "trusted_rows": len(prepared.trusted.y), "test_rows": len(prepared.test.y),
        "preprocessor": prepared.schema.metadata(),
        "note": "Original public dataset CSV files are not redistributed. Split seed is fixed independently of training seeds."
    }
    p.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
