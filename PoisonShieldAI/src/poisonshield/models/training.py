"""Trainable victim and surrogate wrappers, with controlled random seeds."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from .networks import CNN1D, SurrogateMLP


def set_seeds(seed: int, threads: int | None = None) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if threads is not None:
        torch.set_num_threads(max(1, int(threads)))


def resolve_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but no CUDA-enabled GPU is available")
    return torch.device(preference)


def train_surrogate(X: np.ndarray, y: np.ndarray, n_classes: int,
                    *, seed: int, epochs: int, lr: float, device: torch.device,
                    batch_size: int = 256) -> SurrogateMLP:
    set_seeds(seed)
    model = SurrogateMLP(X.shape[1], n_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    model.train()
    for _ in range(int(epochs)):
        for idx in np.array_split(rng.permutation(len(y)), max(1, int(np.ceil(len(y)/batch_size)))):
            if not len(idx):
                continue
            optim.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(X_t[idx]), y_t[idx])
            loss.backward()
            optim.step()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@dataclass
class Victim:
    name: str
    model: object
    device: torch.device

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.name != "cnn":
            return np.asarray(self.model.predict(X), dtype=int)
        result: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for chunk in np.array_split(X, max(1, int(np.ceil(len(X) / 4096)))):
                if len(chunk):
                    pred = self.model(torch.as_tensor(chunk, dtype=torch.float32, device=self.device)).argmax(1)
                    result.append(pred.cpu().numpy())
        return np.concatenate(result)

    def save(self, filename: Path) -> None:
        filename.parent.mkdir(parents=True, exist_ok=True)
        if self.name == "cnn":
            torch.save({"state_dict": self.model.state_dict(), "architecture": "CNN1D"}, filename.with_suffix(".pt"))
        else:
            import joblib
            joblib.dump(self.model, filename.with_suffix(".joblib"))


def train_victim(name: str, X: np.ndarray, y: np.ndarray, n_classes: int,
                 cfg: dict, seed: int, device: torch.device, *, adversarial: bool = False) -> Victim:
    if len(np.unique(y)) < 2:
        raise ValueError("Filtered data contains fewer than two classes; cannot train valid victim")
    set_seeds(seed)
    if name == "rf":
        if adversarial:
            raise ValueError("FGSM adversarial training is only defined for the CNN victim")
        model = RandomForestClassifier(n_estimators=int(cfg.get("rf_trees", 100)),
                                       max_depth=cfg.get("rf_max_depth"),
                                       n_jobs=4, random_state=seed)
        model.fit(X, y)
    elif name == "xgboost":
        if adversarial:
            raise ValueError("FGSM adversarial training is only defined for the CNN victim")
        # The classes may be absent after severe filtering; XGBoost requires contiguous labels.
        observed = np.unique(y)
        if not np.array_equal(observed, np.arange(n_classes)):
            raise ValueError(f"XGBoost requires all classes represented; observed {observed.tolist()}")
        model = XGBClassifier(n_estimators=int(cfg.get("xgb_trees", 100)),
                              max_depth=int(cfg.get("xgb_max_depth", 5)),
                              learning_rate=0.1, subsample=0.9,
                              colsample_bytree=0.9, tree_method="hist", n_jobs=4,
                              objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                              eval_metric="mlogloss" if n_classes > 2 else "logloss", random_state=seed)
        model.fit(X, y)
    elif name == "cnn":
        model = CNN1D(n_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("cnn_learning_rate", 0.001)))
        X_t = torch.as_tensor(X, device=device, dtype=torch.float32)
        y_t = torch.as_tensor(y, device=device, dtype=torch.long)
        rng = np.random.default_rng(seed)
        batch_size = int(cfg.get("cnn_batch_size", 256))
        for _ in range(int(cfg.get("cnn_epochs", 20))):
            model.train()
            for ids in np.array_split(rng.permutation(len(y)), max(1, int(np.ceil(len(y)/batch_size)))):
                opt.zero_grad(set_to_none=True)
                xb, yb = X_t[ids], y_t[ids]
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                if adversarial:
                    # FGSM (model gradient at the current clean batch) is a true adversarial
                    # augmentation; it is not mislabeled as simple Gaussian noise.
                    adv_input = xb.detach().clone().requires_grad_(True)
                    grad_loss = F.cross_entropy(model(adv_input), yb)
                    grad = torch.autograd.grad(grad_loss, adv_input, only_inputs=True)[0]
                    numeric_dim = int(cfg.get("n_numeric_features", xb.shape[1]))
                    # Keep one-hot categorical features fixed under feature-space FGSM.
                    perturb = torch.zeros_like(xb)
                    perturb[:, :numeric_dim] = float(cfg.get("fgsm_epsilon", 0.035)) * grad[:, :numeric_dim].sign()
                    xb_adv = torch.clamp(xb + perturb, 0.0, 1.0)
                    loss = 0.5 * loss + 0.5 * F.cross_entropy(model(xb_adv), yb)
                loss.backward()
                opt.step()
        model.eval()
    else:
        raise ValueError(f"Unknown victim name: {name}")
    return Victim(name, model, device)
