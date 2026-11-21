from copy import deepcopy
from pathlib import Path
import pandas as pd
from poisonshield.pipeline import ExperimentSession, read_config


def make_config(tmp_path):
    cfg = read_config(Path(__file__).resolve().parents[1]/"configs/quick_experiment.yaml")
    cfg = deepcopy(cfg)
    cfg["experiment"].update({"synthetic_rows": 500, "victim":"rf", "write_figures":False,
                               "save_models":True, "include_baselines":False,
                               "output_dir":str(tmp_path)})
    cfg["attack"].update({"epochs":2, "surrogate_epochs":2})
    cfg["defense"].update({"epochs":2, "dbscan_min_samples":5})
    cfg["victims"].update({"rf_trees":12, "xgb_trees":10, "cnn_epochs":2})
    return cfg


def test_full_pipeline_and_outputs(tmp_path):
    session = ExperimentSession(make_config(tmp_path))
    frame = session.run(seed=0, rate=0.1, victim="rf,xgboost,cnn")
    expected = {"clean", "poisoned", "DBSCAN_only", "Discriminator_only", "PoisonShieldAI"}
    assert set(frame.victim) == {"rf", "xgboost", "cnn"}
    assert expected.issubset(set(frame.method))
    assert set(frame.status) == {"ok"}
    assert frame.accuracy.between(0,1).all()
    base = tmp_path / "synthetic_binary_seed0_rate0.1000"
    assert (base/"sanitization_audit.csv.gz").exists()
    assert (base/"evaluation_provenance.csv.gz").exists()
    assert (base/"defended_rf.joblib").exists()
    assert (base/"defended_xgboost.joblib").exists()
    assert (base/"defended_cnn.pt").exists()
    audit = pd.read_csv(base/"sanitization_audit.csv.gz")
    assert "is_injected_poison" not in audit.columns
    assert "stage2_realism_score" in audit.columns


def test_mixed_budget_math(tmp_path):
    cfg = make_config(tmp_path)
    session = ExperimentSession(cfg)
    frame = session.run(seed=1, rate=0.005, victim="rf")
    assert frame.injected_records.unique().tolist() == [1]
