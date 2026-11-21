"""Controlled, offline, classifier-aware GAN poisoning for academic benchmarking."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch.nn import functional as F
from ..models.networks import Generator, Discriminator, SurrogateMLP
from ..models.training import set_seeds, train_surrogate


@dataclass
class AttackResult:
    generator: Generator
    discriminator: Discriminator
    surrogate: SurrogateMLP
    source_class: int
    injection_label: int
    true_source_name: str
    label_policy: str
    training_log: list[dict]
    config: dict


def _choose_source(y: np.ndarray, benign_class: int = 0) -> int:
    valid = y[y != benign_class]
    if not len(valid):
        raise ValueError("Attacker auxiliary split contains no attack-class samples")
    ids, counts = np.unique(valid, return_counts=True)
    return int(ids[counts.argmax()])


def train_attack(X: np.ndarray, y: np.ndarray, schema, classes: list[str],
                 cfg: dict, seed: int, device: torch.device) -> AttackResult:
    set_seeds(seed)
    n_classes = len(classes)
    source = _choose_source(y)
    policy = str(cfg.get("label_policy", "dirty_target_benign"))
    if policy not in {"dirty_target_benign", "clean_label"}:
        raise ValueError("Attack label_policy must be dirty_target_benign or clean_label")
    target = 0 if policy == "dirty_target_benign" else source
    surrogate = train_surrogate(X, y, n_classes, seed=seed, epochs=int(cfg.get("surrogate_epochs", 20)),
                                lr=float(cfg.get("surrogate_learning_rate", 0.001)), device=device)
    n_numeric = len(schema.numeric_indices)
    categories = [len(group) for group in schema.categorical_groups]
    G = Generator(int(cfg.get("z_dim", 64)), n_numeric, categories).to(device)
    D = Discriminator(X.shape[1]).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=float(cfg.get("learning_rate", 0.0002)), betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=float(cfg.get("learning_rate", 0.0002)), betas=(0.5, 0.999))
    X_source = X[y == source]
    rng = np.random.default_rng(seed + 521)
    max_rows = cfg.get("max_train_rows")
    if max_rows and len(X_source) > int(max_rows):
        X_source = X_source[rng.choice(len(X_source), int(max_rows), replace=False)]
    real = torch.as_tensor(X_source, dtype=torch.float32, device=device)
    batch_size = int(cfg.get("batch_size", 64))
    log: list[dict] = []
    temperature = float(cfg.get("categorical_temperature", 0.75))
    weight = float(cfg.get("lambda_weight", 0.5))
    for epoch in range(int(cfg.get("epochs", 500))):
        order = rng.permutation(len(real))
        d_loss_running, g_loss_running, score_running, n_batches = 0., 0., 0., 0
        for ids in np.array_split(order, max(1, math.ceil(len(order) / batch_size))):
            actual_bs = len(ids)
            x_real = real[ids]
            for _ in range(int(cfg.get("discriminator_steps", 2))):
                z = torch.randn(actual_bs, G.z_dim, device=device)
                x_fake = G(z, temperature=temperature, stochastic=True).detach()
                opt_d.zero_grad(set_to_none=True)
                d_loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(D(x_real), torch.ones(actual_bs, device=device)) +
                    F.binary_cross_entropy_with_logits(D(x_fake), torch.zeros(actual_bs, device=device)))
                d_loss.backward()
                opt_d.step()
            for param in D.parameters():
                param.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            z = torch.randn(actual_bs, G.z_dim, device=device)
            fake = G(z, temperature=temperature, stochastic=True)
            adv_realism = F.binary_cross_entropy_with_logits(D(fake), torch.ones(actual_bs, device=device))
            # A dirty-label sample is truly attack-like, but injected with a benign label.
            # Maximising surrogate loss for the injected label favours conflicts with that label.
            assigned_labels = torch.full((actual_bs,), target, device=device, dtype=torch.long)
            surrogate_loss = F.cross_entropy(surrogate(fake), assigned_labels)
            loss = adv_realism - weight * surrogate_loss
            loss.backward()
            opt_g.step()
            for param in D.parameters():
                param.requires_grad_(True)
            d_loss_running += float(d_loss.detach())
            g_loss_running += float(loss.detach())
            with torch.no_grad():
                score_running += float(torch.sigmoid(D(fake.detach())).mean())
            n_batches += 1
        log.append({"epoch": epoch + 1, "d_loss": d_loss_running / n_batches,
                    "g_loss": g_loss_running / n_batches, "attacker_realism_score": score_running / n_batches})
    G.eval(); D.eval()
    return AttackResult(G, D, surrogate, source, target, classes[source], policy, log, dict(cfg))


def generate_poison(attack: AttackResult, n_samples: int, seed: int,
                    device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_samples < 0:
        raise ValueError("Requested negative number of poisons")
    set_seeds(seed)
    generated: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n_samples, 2048):
            n = min(2048, n_samples-start)
            z = torch.randn(n, attack.generator.z_dim, device=device)
            output = attack.generator(z, stochastic=False)
            generated.append(output.cpu().numpy().astype(np.float32))
    n_dim = attack.generator.n_numeric + sum(attack.generator.categorical_sizes)
    X = np.concatenate(generated) if generated else np.empty((0, n_dim), dtype=np.float32)
    y = np.full(n_samples, attack.injection_label, dtype=np.int64)
    ids = np.asarray([f"poison:{seed}:{i}" for i in range(n_samples)], dtype=str)
    return X, y, ids
