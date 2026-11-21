import numpy as np
import pandas as pd
from poisonshield.data.loaders import synthetic_flows, identify_labels
from poisonshield.data.preprocessing import prepare_dataset, FeaturePreprocessor


def test_synthetic_split_disjoint_and_shapes():
    raw = synthetic_flows(500, seed=0)
    d = prepare_dataset(raw, "synthetic", "binary", split_seed=0)
    all_ids = np.concatenate([d.attacker.ids, d.victim.ids, d.trusted.ids, d.test.ids])
    assert len(set(all_ids)) == len(all_ids)
    assert sum(len(v.y) for v in [d.attacker, d.victim, d.trusted, d.test]) == 500
    for part in [d.attacker, d.victim, d.trusted, d.test]:
        assert part.X.shape == (len(part.y), d.schema.n_features)
        assert np.isfinite(part.X).all()
        assert (part.X >= 0).all() and (part.X <= 1).all()
    assert len(d.schema.categorical_groups) == 2
    assert "Source IP" in d.schema.dropped_columns
    assert "Timestamp" in d.schema.dropped_columns


def test_rare_and_unseen_category_and_test_extremes():
    frame = pd.DataFrame({"number": list(range(2000)), "proto": ["tcp"]*1990+["udp"]*9+["icmp"],
                          "Source IP": ["127.0.0.1"]*2000})
    p = FeaturePreprocessor().fit(frame)
    transformed = p.transform(pd.DataFrame({"number": [10000], "proto": ["nonexistent"],
                                              "Source IP": ["9.9.9.9"]}))
    assert transformed[0, 0] == 1.0  # Clipped using TRAIN-ONLY min/max.
    assert np.isclose(transformed[0, p.categorical_groups[0]].sum(), 1.)
    assert transformed[0, p.categorical_groups[0][-1]] == 1.0


def test_label_leakage_and_protocol_categorical():
    raw = pd.DataFrame({"Protocol": [6, 17, 1, 6], "num": [1., 2., 3., 4.],
                        "type": ["normal", "dos", "recon", "normal"], "label": [0, 1, 1, 0]})
    X, y, labels = identify_labels(raw, "toniot", "multiclass")
    assert "type" not in X.columns and "label" not in X.columns
    assert labels[0] == "normal"
    feat = FeaturePreprocessor().fit(X)
    assert "Protocol" in feat.categorical_cols
    assert feat.n_features == 4


def test_multiclass_has_distinct_labels():
    d = prepare_dataset(synthetic_flows(500), "synthetic", "multiclass")
    assert d.classes == ["normal", "dos", "recon"]
    assert set(d.attacker.y) == {0, 1, 2}
