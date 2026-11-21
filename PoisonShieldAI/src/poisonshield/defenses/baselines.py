"""Optional transparent defence baselines; all operate on locally stored features."""
from __future__ import annotations
import numpy as np
import torch
from scipy.special import softmax
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from ..models.training import train_surrogate


def _retain_after_ranking(scores: np.ndarray, n_reject: int) -> np.ndarray:
    """High anomaly/conflict scores are rejected; selection never uses poison truth."""
    keep = np.ones(len(scores), dtype=bool)
    if n_reject <= 0:
        return keep
    n_reject = min(n_reject, len(scores)-1)
    remove = np.argsort(np.nan_to_num(scores, nan=-np.inf), kind="stable")[-n_reject:]
    keep[remove] = False
    return keep


def spectral_signatures(X: np.ndarray, y: np.ndarray, n_reject: int, seed: int) -> np.ndarray:
    scores = np.zeros(len(X), dtype=float)
    for cls in np.unique(y):
        ids = np.flatnonzero(y == cls)
        if len(ids) < 3:
            continue
        centered = X[ids].astype(np.float64) - np.mean(X[ids], axis=0)
        spectral = TruncatedSVD(n_components=1, random_state=seed).fit(centered)
        projection = centered @ spectral.components_[0]
        scores[ids] = projection ** 2 / (float(np.var(projection)) + 1e-9)
    return _retain_after_ranking(scores, n_reject)


def gradient_influence_proxy(X: np.ndarray, y: np.ndarray, trust_X: np.ndarray,
                             trust_y: np.ndarray, n_classes: int, n_reject: int,
                             seed: int) -> np.ndarray:
    """Gradient-conflict proxy, NOT exact Hessian-inverse influence functions.

    Fit a reference softmax/logistic model on trusted train records, calculate
    the gradient of held-out trusted CE loss, and remove training samples whose
    per-example CE gradient has the strongest opposing alignment.
    """
    idx = np.arange(len(trust_y))
    class_counts = np.bincount(trust_y)
    class_counts = class_counts[class_counts > 0]
    strat = trust_y if len(class_counts) >= 2 and class_counts.min() >= 2 else None
    train, validate = train_test_split(idx, test_size=0.3, random_state=seed, stratify=strat)
    if len(np.unique(trust_y[train])) < 2:
        return np.ones(len(y), dtype=bool)
    clf = LogisticRegression(max_iter=350, random_state=seed)
    clf.fit(trust_X[train], trust_y[train])

    def aligned_probabilities(x: np.ndarray) -> np.ndarray:
        raw = clf.predict_proba(x)
        result = np.full((len(x), n_classes), 1e-8, dtype=float)
        result[:, clf.classes_.astype(int)] = raw
        return result / result.sum(axis=1, keepdims=True)

    trust_pred = aligned_probabilities(trust_X[validate])
    trusted_onehot = np.eye(n_classes)[trust_y[validate]]
    trusted_extended = np.column_stack([trust_X[validate], np.ones(len(validate))])
    val_gradient = ((trust_pred - trusted_onehot).T @ trusted_extended) / len(validate)
    scores = np.empty(len(X), dtype=float)
    for start in range(0, len(X), 8192):
        end = min(len(X), start+8192)
        predicted = aligned_probabilities(X[start:end])
        residual = predicted - np.eye(n_classes)[y[start:end]]
        extended = np.column_stack([X[start:end], np.ones(end-start)])
        scores[start:end] = -np.sum((residual @ val_gradient) * extended, axis=1)
    return _retain_after_ranking(scores, n_reject)


def deep_knn(X: np.ndarray, y: np.ndarray, trust_X: np.ndarray,
             trust_y: np.ndarray, n_classes: int, n_reject: int,
             seed: int, device: torch.device) -> np.ndarray:
    """k-NN label-consistency in learned neural embeddings (not raw feature k-NN)."""
    encoder = train_surrogate(trust_X, trust_y, n_classes, seed=seed+200,
                              epochs=15, lr=0.001, device=device)

    def embed(v: np.ndarray) -> np.ndarray:
        chunks = []
        with torch.no_grad():
            for start in range(0, len(v), 8192):
                xt = torch.as_tensor(v[start:start+8192], dtype=torch.float32, device=device)
                chunks.append(encoder.embedding(xt).cpu().numpy())
        return np.concatenate(chunks)

    k = min(7, len(trust_X))
    nn = NearestNeighbors(n_neighbors=k, n_jobs=4).fit(embed(trust_X))
    embeddings = embed(X)
    scores = np.empty(len(X), dtype=float)
    for start in range(0, len(X), 8192):
        end = min(len(X), start+8192)
        distances, ids = nn.kneighbors(embeddings[start:end])
        agreement = (trust_y[ids] == y[start:end, None]).mean(axis=1)
        scores[start:end] = 1.0 - agreement + 0.03 * distances.mean(axis=1)
    return _retain_after_ranking(scores, n_reject)


def run_baseline(method: str, X: np.ndarray, y: np.ndarray,
                 trust_X: np.ndarray, trust_y: np.ndarray, n_classes: int,
                 n_reject: int, seed: int, device: torch.device) -> np.ndarray:
    if method == "spectral":
        return spectral_signatures(X, y, n_reject, seed)
    if method == "gradient_influence_proxy":
        return gradient_influence_proxy(X, y, trust_X, trust_y, n_classes, n_reject, seed)
    if method == "deep_knn":
        return deep_knn(X, y, trust_X, trust_y, n_classes, n_reject, seed, device)
    raise ValueError(f"Unknown baseline: {method}")
