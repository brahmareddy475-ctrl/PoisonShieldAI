import numpy as np
import torch
from poisonshield.data.loaders import synthetic_flows
from poisonshield.data.preprocessing import prepare_dataset
from poisonshield.attacks.gan import train_attack, generate_poison
from poisonshield.defenses.filters import calibrate_threshold, density_filter, train_defender, apply_defense


def test_attack_generation_onehot_and_labels():
    d = prepare_dataset(synthetic_flows(400), "synthetic", "binary")
    cfg = {"z_dim": 16, "batch_size": 32, "epochs": 2, "surrogate_epochs": 2,
           "learning_rate": 0.0002, "surrogate_learning_rate": 0.001,
           "lambda_weight": 0.5, "discriminator_steps": 2, "label_policy":"dirty_target_benign"}
    device = torch.device("cpu")
    attack = train_attack(d.attacker.X, d.attacker.y, d.schema, d.classes, cfg, 0, device)
    X, y, ids = generate_poison(attack, 37, 200, device)
    assert X.shape == (37, d.schema.n_features)
    assert set(y) == {0} and len(set(ids)) == 37
    assert (X >= 0).all() and (X <= 1).all()
    for group in d.schema.categorical_groups:
        assert np.allclose(X[:, group].sum(1), 1.)
        assert np.all((X[:, group] == 0) | (X[:, group] == 1))


def test_calibration_rejection_limit():
    scores = np.arange(100) / 100
    threshold, rejected = calibrate_threshold(scores, 0.05)
    assert rejected <= 0.05
    assert np.count_nonzero(scores < threshold) <= 5
    threshold2, rejected2 = calibrate_threshold(np.ones(16), 0.05)
    assert rejected2 == 0


def test_defender_and_sequential_audit():
    d = prepare_dataset(synthetic_flows(500), "synthetic", "binary")
    device = torch.device("cpu")
    cfg = {"z_dim": 16, "epochs": 2, "batch_size": 32, "learning_rate": 0.0002,
           "max_reject_clean":0.05, "trusted_calibration_fraction":0.2,
           "dbscan_mode":"exact", "dbscan_exact_max_rows": 1500,
           "dbscan_min_samples": 5, "dbscan_eps_quantile":0.95}
    defender = train_defender(d.trusted.X, d.trusted.y, d.schema, cfg, seed=0, device=device)
    assert defender.calibration_rejected <= 0.05
    X = d.victim.X
    result = apply_defense(X, d.victim.ids, d.trusted.X, defender, cfg)
    assert len(result.audit) == len(X)
    assert np.array_equal(result.final_keep, result.stage1.retained & (result.discriminator_scores >= defender.threshold))
    assert not result.audit.sample_id.duplicated().any()
    assert set(result.audit.decision_reason.unique()).issubset({"accepted", "stage1_density", "stage2_discriminator"})


def test_approx_mode_is_explicit():
    d = prepare_dataset(synthetic_flows(400), "synthetic", "binary")
    result = density_filter(d.victim.X, d.trusted.X, min_samples=5,
                            mode="auto", exact_max_rows=5)
    assert result.mode_used == "reference_knn"
    assert result.retained.shape == (len(d.victim.y),)
