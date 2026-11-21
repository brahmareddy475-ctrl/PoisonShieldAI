"""PoisonShieldAI stages: density filtering followed by independent defender GAN."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from ..models.networks import Generator, Discriminator
from ..models.training import set_seeds


@dataclass
class Defender:
    generator: Generator
    discriminator: Discriminator
    threshold: float
    calibration_rejected: float
    calibration_count: int
    calibration_scores: np.ndarray
    training_log: list[dict]
    device: torch.device

    def score(self, X: np.ndarray) -> np.ndarray:
        scores = []
        self.discriminator.eval()
        with torch.no_grad():
            for start in range(0, len(X), 8192):
                xt = torch.as_tensor(X[start:start+8192], device=self.device, dtype=torch.float32)
                scores.append(torch.sigmoid(self.discriminator(xt)).cpu().numpy())
        return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)


@dataclass
class DensityResult:
    retained: np.ndarray
    neighbor_count: np.ndarray
    kth_distance: np.ndarray
    epsilon: float
    min_samples: int
    mode_used: str


@dataclass
class DefenseResult:
    stage1: DensityResult
    final_keep: np.ndarray
    discriminator_scores: np.ndarray
    threshold: float
    audit: pd.DataFrame


def calibrate_threshold(scores: np.ndarray, max_reject: float = 0.05) -> tuple[float, float]:
    """Select an empirical order statistic with <= floor(alpha*n) clean rejects."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration requires non-empty, finite 1D scores")
    if not 0 <= max_reject < 1:
        raise ValueError("max_reject must be in [0,1)")
    k = min(len(scores) - 1, int(math.floor(max_reject * len(scores))))
    threshold = float(np.sort(scores)[k])
    reject_fraction = float((scores < threshold).mean())
    assert reject_fraction <= max_reject + 1e-12
    return threshold, reject_fraction


def train_defender(trusted_X: np.ndarray, trusted_y: np.ndarray, schema,
                   cfg: dict, seed: int, device: torch.device) -> Defender:
    """Trains only from defender trusted samples; no attacker object is accepted."""
    if len(trusted_X) < 30:
        raise ValueError("A minimum of 30 trusted records is required")
    set_seeds(seed + 100)
    rng = np.random.default_rng(seed + 100)
    index = np.arange(len(trusted_y))
    calibration_frac = float(cfg.get("trusted_calibration_fraction", 0.2))
    if not 0 < calibration_frac < 0.8:
        raise ValueError("Defender calibration fraction must lie between 0 and 0.8")
    counts = np.bincount(trusted_y)
    counts = counts[counts > 0]
    stratify = trusted_y if len(counts) >= 2 and counts.min() >= 2 and int(len(index)*calibration_frac) >= len(counts) else None
    train_idx, calibrate_idx = train_test_split(index, test_size=calibration_frac,
                                               random_state=seed + 100, stratify=stratify)
    source_data = trusted_X[train_idx]
    max_rows = cfg.get("max_train_rows")
    if max_rows and len(source_data) > int(max_rows):
        source_data = source_data[rng.choice(len(source_data), int(max_rows), replace=False)]
    real = torch.as_tensor(source_data, dtype=torch.float32, device=device)
    z_dim = int(cfg.get("z_dim", 64))
    G = Generator(z_dim, len(schema.numeric_indices), [len(v) for v in schema.categorical_groups]).to(device)
    D = Discriminator(trusted_X.shape[1]).to(device)
    optimizer_g = torch.optim.Adam(G.parameters(), lr=float(cfg.get("learning_rate", 0.0002)), betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(D.parameters(), lr=float(cfg.get("learning_rate", 0.0002)), betas=(0.5, 0.999))
    bs = int(cfg.get("batch_size", 64))
    training_log = []
    for epoch in range(int(cfg.get("epochs", 500))):
        loss_d_total, loss_g_total, n_batches = 0., 0., 0
        for indices in np.array_split(rng.permutation(len(real)), max(1, math.ceil(len(real) / bs))):
            rb = real[indices]
            n = len(indices)
            fake = G(torch.randn(n, z_dim, device=device)).detach()
            optimizer_d.zero_grad(set_to_none=True)
            loss_d = 0.5 * (
                F.binary_cross_entropy_with_logits(D(rb), torch.ones(n, device=device)) +
                F.binary_cross_entropy_with_logits(D(fake), torch.zeros(n, device=device)))
            loss_d.backward(); optimizer_d.step()
            for p in D.parameters(): p.requires_grad_(False)
            optimizer_g.zero_grad(set_to_none=True)
            fake2 = G(torch.randn(n, z_dim, device=device))
            loss_g = F.binary_cross_entropy_with_logits(D(fake2), torch.ones(n, device=device))
            loss_g.backward(); optimizer_g.step()
            for p in D.parameters(): p.requires_grad_(True)
            loss_d_total += float(loss_d.detach()); loss_g_total += float(loss_g.detach()); n_batches += 1
        training_log.append({"epoch": epoch+1, "d_loss":loss_d_total/n_batches,
                             "g_loss": loss_g_total/n_batches})
    G.eval(); D.eval()
    with torch.no_grad():
        scores = []
        for start in range(0, len(calibrate_idx), 8192):
            tensor = torch.as_tensor(trusted_X[calibrate_idx[start:start+8192]], dtype=torch.float32, device=device)
            scores.append(torch.sigmoid(D(tensor)).cpu().numpy())
        calibration_scores = np.concatenate(scores)
    tau, fraction = calibrate_threshold(calibration_scores, float(cfg.get("max_reject_clean", 0.05)))
    return Defender(G, D, tau, fraction, len(calibrate_idx), calibration_scores, training_log, device)


def density_filter(X: np.ndarray, trusted_X: np.ndarray, *, min_samples: int = 10,
                   eps_quantile: float = 0.95, mode: str = "auto", exact_max_rows: int = 30000,
                   audit_neighbors: int = 24) -> DensityResult:
    if min_samples < 2 or len(trusted_X) < min_samples:
        raise ValueError("Not enough trusted records to calibrate DBSCAN min_samples")
    if not 0 < eps_quantile < 1:
        raise ValueError("eps_quantile must lie in (0,1)")
    trust_nn = NearestNeighbors(n_neighbors=min_samples, algorithm="auto", n_jobs=4)
    trust_nn.fit(trusted_X)
    trust_dist, _ = trust_nn.kneighbors(trusted_X)
    eps = max(float(np.quantile(trust_dist[:, -1], eps_quantile)), 1e-6)
    if mode not in {"auto", "exact", "reference_knn"}:
        raise ValueError("DBSCAN mode must be auto, exact, or reference_knn")
    requested = mode
    if mode == "auto":
        mode = "exact" if len(X) <= exact_max_rows else "reference_knn"
    if mode == "exact" and len(X) > exact_max_rows:
        raise MemoryError(f"Exact DBSCAN on {len(X)} rows exceeds safe limit {exact_max_rows}. "
                          f"Use dbscan_mode: auto or reference_knn, or raise cap if you have enough RAM.")
    if mode == "exact":
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=4).fit_predict(X)
        keep = labels != -1
        k = min(max(audit_neighbors, min_samples), len(X))
        neighbors = NearestNeighbors(n_neighbors=k, n_jobs=4).fit(X)
        distances, _ = neighbors.kneighbors(X)
        counts = (distances <= eps).sum(axis=1).astype(int)
        kth = distances[:, min(min_samples, k)-1].astype(float)
    else:
        # Scalable approximation: distance to the kth trusted neighbor, NOT full DBSCAN.
        # The CSV reports mode_used explicitly. It is never silently called exact DBSCAN.
        k = min(max(audit_neighbors, min_samples), len(trusted_X))
        neighbors = NearestNeighbors(n_neighbors=k, n_jobs=4).fit(trusted_X)
        counts = np.empty(len(X), dtype=int)
        kth = np.empty(len(X), dtype=float)
        keep = np.empty(len(X), dtype=bool)
        for start in range(0, len(X), 4096):
            ds, _ = neighbors.kneighbors(X[start:start+4096])
            kth[start:start+4096] = ds[:, min_samples-1]
            counts[start:start+4096] = (ds <= eps).sum(axis=1)
            keep[start:start+4096] = ds[:, min_samples-1] <= eps
    return DensityResult(keep, counts, kth, eps, min_samples, mode)


def apply_defense(X: np.ndarray, sample_ids: np.ndarray, trusted_X: np.ndarray,
                  defender: Defender, cfg: dict) -> DefenseResult:
    """Sequential filter only. Poison indicators and training labels are never inputs."""
    if len(X) != len(sample_ids):
        raise ValueError("X and sample_ids must match")
    stage1 = density_filter(
        X, trusted_X, min_samples=int(cfg.get("dbscan_min_samples", 10)),
        eps_quantile=float(cfg.get("dbscan_eps_quantile", 0.95)),
        mode=str(cfg.get("dbscan_mode", "auto")),
        exact_max_rows=int(cfg.get("dbscan_exact_max_rows", 30000)),
        audit_neighbors=int(cfg.get("max_neighbors_for_audit", 24))
    )
    scores = np.full(len(X), np.nan, dtype=np.float64)
    scores[stage1.retained] = defender.score(X[stage1.retained])
    final = stage1.retained & (scores >= defender.threshold)
    reason = np.where(~stage1.retained, "stage1_density", np.where(~final, "stage2_discriminator", "accepted"))
    audit = pd.DataFrame({
        "sample_id": sample_ids, "stage1_keep": stage1.retained,
        "stage1_neighbor_count_capped": stage1.neighbor_count,
        "stage1_kth_neighbor_distance": stage1.kth_distance,
        "stage1_epsilon": stage1.epsilon,
        "stage1_mode": stage1.mode_used,
        "stage2_realism_score": scores,
        "stage2_threshold": defender.threshold,
        "final_keep": final,
        "decision_reason": reason,
    })
    return DefenseResult(stage1, final, scores, defender.threshold, audit)
