from __future__ import annotations
import copy
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import yaml
from .data.loaders import load_raw
from .data.preprocessing import PreparedData, prepare_dataset, save_manifest
from .attacks.gan import AttackResult, train_attack, generate_poison
from .defenses.filters import Defender, apply_defense, calibrate_threshold, train_defender
from .defenses.baselines import run_baseline
from .models.training import resolve_device, set_seeds, train_victim
from .evaluation.metrics import classification_metrics, filter_metrics
from .evaluation.statistics import summarize, paired_tests
from .evaluation.plots import plot_run, plot_sweep


def deep_merge(base: dict, overrides: dict) -> dict:
    result = copy.deepcopy(base)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def read_config(path: str | Path, extra: str | Path | None = None) -> dict:
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"YAML root must be a map in {path}")
    if extra:
        changes = yaml.safe_load(Path(extra).read_text(encoding="utf-8"))
        config = deep_merge(config, changes)
    required = ["experiment", "attack", "defense", "victims"]
    if any(k not in config for k in required):
        raise ValueError(f"Configuration requires sections {required}; start with configs/quick_experiment.yaml")
    return config


def _serialize(data: Any) -> Any:
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, (np.integer, np.floating, np.bool_)):
        return data.item()
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, torch.device):
        return str(data)
    if isinstance(data, dict):
        return {str(k): _serialize(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_serialize(v) for v in data]
    return data


def _write_json(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_serialize(values), indent=2, allow_nan=False), encoding="utf-8")


def _valid_models(selection: str) -> list[str]:
    if selection == "all":
        return ["rf", "xgboost", "cnn"]
    names = [s.strip() for s in selection.split(",")]
    if any(name not in {"rf", "xgboost", "cnn"} for name in names):
        raise ValueError("victim must be 'all', 'rf', 'xgboost', or 'cnn' (comma separated allowed)")
    return names


class ExperimentSession:
    def __init__(self, config: dict):
        self.cfg = copy.deepcopy(config)
        ex = self.cfg["experiment"]
        self.dataset = str(ex["dataset"])
        self.task = str(ex.get("task", "binary"))
        self.device = resolve_device(str(ex.get("device", "auto")))
        threads = int(ex.get("torch_threads", 2))
        set_seeds(0, threads=threads)
        raw, paths = load_raw(
            self.dataset, ex.get("data_path"), max_rows_per_file=ex.get("max_rows_per_file"),
            synthetic_rows=int(ex.get("synthetic_rows", 850)),
            synthetic_seed=int(ex.get("random_split_seed", 0)))
        self.data = prepare_dataset(raw, self.dataset, self.task,
                                    split_seed=int(ex.get("random_split_seed", 0)), source_names=paths)
        self.output = Path(str(ex.get("output_dir", "outputs"))).expanduser()
        self.output.mkdir(parents=True, exist_ok=True)
        save_manifest(self.data, self.output / "dataset_manifest.json", dataset=self.dataset)
        _write_json(self.output / "effective_config.json", self.cfg)
        self.attack_cache: tuple[int, AttackResult] | None = None
        self.defender_cache: tuple[int, Defender] | None = None
        self.clean_model_cache: dict[tuple[int, str], object] = {}
        ids = [self.data.attacker.ids, self.data.victim.ids, self.data.trusted.ids, self.data.test.ids]
        if len(set(np.concatenate(ids))) != sum(map(len, ids)):
            raise AssertionError("Data split leakage: overlapping sample IDs")

    def _models(self, seed: int) -> tuple[AttackResult, Defender]:
        if self.attack_cache is None or self.attack_cache[0] != seed:
            self.attack_cache = (seed, train_attack(
                self.data.attacker.X, self.data.attacker.y, self.data.schema,
                self.data.classes, self.cfg["attack"], seed, self.device))
        if self.defender_cache is None or self.defender_cache[0] != seed:
            dcfg = dict(self.cfg["defense"])
            dcfg["z_dim"] = int(self.cfg["attack"].get("z_dim", 64))
            self.defender_cache = (seed, train_defender(
                self.data.trusted.X, self.data.trusted.y, self.data.schema,
                dcfg, seed, self.device))
        return self.attack_cache[1], self.defender_cache[1]

    def run(self, *, seed: int | None = None, rate: float | None = None,
            victim: str | None = None) -> pd.DataFrame:
        ex = self.cfg["experiment"]
        seed = int(ex.get("seed", 0) if seed is None else seed)
        rate = float(ex.get("poison_rate", 0.1) if rate is None else rate)
        selection = str(ex.get("victim", "all") if victim is None else victim)
        if not 0 < rate <= 0.2:
            raise ValueError("Poisoning budget must be >0 and <=0.20")
        models = _valid_models(selection)
        attack, defender = self._models(seed)
        d = self.data
        n_poison = int(math.floor(rate * len(d.victim.y)))
        if n_poison < 1:
            raise ValueError(f"Budget {rate} yields zero poisons for {len(d.victim.y)} victim records; "
                             "use a larger dataset or poisoning budget; budget is never silently exceeded")
        poison_x, poison_y, poison_ids = generate_poison(attack, n_poison, seed + 7000, self.device)
        X = np.concatenate([d.victim.X, poison_x], axis=0)
        y = np.concatenate([d.victim.y, poison_y], axis=0)
        ids = np.concatenate([d.victim.ids, poison_ids])
        is_poison = np.concatenate([np.zeros(len(d.victim.y), bool), np.ones(n_poison, bool)])
        defense = apply_defense(X, ids, d.trusted.X, defender, self.cfg["defense"])
        stage1 = defense.stage1.retained
        stage2 = defense.discriminator_scores
        threshold = defender.threshold
        # Independent Stage 2 ablation applies the frozen defender to all mixed records.
        discriminator_only = defender.score(X) >= threshold
        all_keep = np.ones(len(y), dtype=bool)
        candidate_filters: dict[str, np.ndarray] = {
            "poisoned": all_keep, "DBSCAN_only": stage1,
            "Discriminator_only": discriminator_only, "PoisonShieldAI": defense.final_keep
        }
        if bool(ex.get("include_baselines", True)):
            # All budget-aware baselines get only the declared attack budget,
            # never sample-wise poisoning provenance.
            budget = min(len(y)-1, round(n_poison * float(self.cfg["victims"].get("baseline_rejection_multiplier", 1.0))))
            for name in ex.get("baseline_methods", ["spectral", "gradient_influence_proxy", "deep_knn"]):
                candidate_filters[name] = run_baseline(
                    name, X, y, d.trusted.X, d.trusted.y, len(d.classes), budget, seed, self.device)
            if bool(ex.get("include_oracle", False)):
                # This is a NON-DEPLOYABLE, knowingly attacker-informed reference.
                attacker_scores = []
                with torch.no_grad():
                    for start in range(0, len(d.trusted.y), 8192):
                        z = torch.as_tensor(d.trusted.X[start:start+8192], device=self.device, dtype=torch.float32)
                        attacker_scores.append(torch.sigmoid(attack.discriminator(z)).cpu().numpy())
                oracle_tau, _ = calibrate_threshold(np.concatenate(attacker_scores),
                      float(self.cfg["defense"].get("max_reject_clean", 0.05)))
                with torch.no_grad():
                    output = []
                    for start in range(0, len(X), 8192):
                        xt = torch.as_tensor(X[start:start+8192], device=self.device, dtype=torch.float32)
                        output.append(torch.sigmoid(attack.discriminator(xt)).cpu().numpy())
                candidate_filters["attacker_discriminator_oracle"] = np.concatenate(output) >= oracle_tau

        name = f"{self.dataset}_{self.task}_seed{seed}_rate{rate:.4f}"
        run_dir = self.output / name
        run_dir.mkdir(parents=True, exist_ok=True)
        provenance = pd.DataFrame({"sample_id": ids, "is_injected_poison": is_poison,
                                   "injected_label": y})
        # Real defender receives neither the provenance dataframe nor any poison mask.
        provenance.to_csv(run_dir / "evaluation_provenance.csv.gz", index=False, compression="gzip")
        defense.audit.to_csv(run_dir / "sanitization_audit.csv.gz", index=False, compression="gzip")
        _write_json(run_dir / "attack_defense_details.json", {
            "dataset": self.dataset, "task": self.task, "seed": seed, "nominal_poison_budget": rate,
            "poison_count": n_poison, "effective_poison_fraction": n_poison/len(X),
            "source_attack_class": d.classes[attack.source_class],
            "assigned_injection_label": d.classes[attack.injection_label],
            "attack_label_policy": attack.label_policy,
            "lambda_weight": self.cfg["attack"].get("lambda_weight", 0.5),
            "stage1_epsilon": defense.stage1.epsilon,
            "stage1_mode": defense.stage1.mode_used,
            "stage1_min_samples": defense.stage1.min_samples,
            "stage2_threshold": threshold,
            "trusted_calibration_count": defender.calibration_count,
            "trusted_calibration_rejection": defender.calibration_rejected,
            "combined_clean_retention_on_victim": float(defense.final_keep[~is_poison].mean()),
            "note": "Injected origin labels are available ONLY to evaluation harness; defender independently trained on trusted data."
        })
        pd.DataFrame(attack.training_log).to_csv(run_dir / "attacker_gan_training.csv", index=False)
        pd.DataFrame(defender.training_log).to_csv(run_dir / "defender_gan_training.csv", index=False)
        rows: list[dict] = []
        diagnostic = {}
        for model_name in models:
            baseline_model = self.clean_model_cache.get((seed, model_name))
            if baseline_model is None:
                baseline_model = train_victim(model_name, d.victim.X, d.victim.y, len(d.classes),
                                             (self.cfg["victims"] | {"n_numeric_features": len(d.schema.numeric_indices)}), seed, self.device)
                self.clean_model_cache[(seed, model_name)] = baseline_model
            all_methods = {"clean": None} | candidate_filters
            if model_name == "cnn" and bool(ex.get("include_baselines", True)):
                all_methods["FGSM_adversarial_training"] = all_keep
            for method, keep in all_methods.items():
                row: dict[str, Any] = {
                    "dataset": self.dataset, "task": self.task, "seed": seed,
                    "poison_rate": rate, "victim": model_name, "method": method,
                    "test_records": len(d.test.y), "clean_victim_records": len(d.victim.y),
                    "injected_records": n_poison, "status": "ok", "error": ""
                }
                if method == "clean":
                    model = baseline_model
                    data_for_eval = d.victim.X
                    keep_for_metric = None
                else:
                    data_for_eval = X[keep]
                    keep_for_metric = keep
                    y_for_training = y[keep]
                    if len(np.unique(y_for_training)) < len(d.classes):
                        row["status"] = "insufficient_classes"
                        row["error"] = "Filtering eliminated one or more classes; no model was trained"
                        rows.append(row)
                        continue
                    try:
                        model = train_victim(model_name, data_for_eval, y_for_training,
                            len(d.classes), (self.cfg["victims"] | {"n_numeric_features": len(d.schema.numeric_indices)}), seed, self.device,
                            adversarial=(method == "FGSM_adversarial_training"))
                    except (ValueError, RuntimeError) as exc:
                        row["status"] = "training_failed"
                        row["error"] = str(exc)
                        rows.append(row)
                        continue
                predictions = model.predict(d.test.X)
                details = classification_metrics(d.test.y, predictions, len(d.classes))
                diagnostic[f"{model_name}/{method}"] = {"confusion_matrix": details.pop("confusion_matrix")}
                row.update(details)
                if keep_for_metric is None:
                    row.update({"clean_retention": 1., "poison_removal": np.nan,
                        "poison_detection_recall": np.nan, "poison_detection_precision": np.nan,
                        "poison_detection_f1": np.nan, "filter_false_positive_rate": np.nan,
                        "filtered_rows": len(d.victim.y), "removed_clean": 0, "removed_poison": 0, "retained_poison": 0})
                else:
                    row.update(filter_metrics(keep_for_metric, is_poison))
                if method == "PoisonShieldAI" and bool(ex.get("save_models", True)):
                    model.save(run_dir / f"defended_{model_name}")
                rows.append(row)
        metrics = pd.DataFrame(rows)
        for model_name in models:
            select = metrics[(metrics.victim == model_name) & (metrics.status == "ok")]
            clean = select[select.method == "clean"]
            poisoned = select[select.method == "poisoned"]
            if not clean.empty and not poisoned.empty:
                attack_drop = float(clean.iloc[0].macro_f1 - poisoned.iloc[0].macro_f1)
                idx = metrics.victim == model_name
                metrics.loc[idx, "attack_drop_macro_f1"] = attack_drop
                if attack_drop > 1e-12:
                    metrics.loc[idx, "recovery_fraction_macro_f1"] = (
                        metrics.loc[idx, "macro_f1"] - poisoned.iloc[0].macro_f1) / attack_drop
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        _write_json(run_dir / "confusion_matrices.json", {"class_names": d.classes, "matrices": diagnostic})
        if bool(ex.get("write_figures", True)):
            plot_run(metrics, run_dir / "figures")
        return metrics


def run_sweep(config: dict, datasets: list[str], victims: str, seeds: list[int],
              rates: list[float], data_path: str | None = None) -> pd.DataFrame:
    """Caches preprocessed dataset, attack/defender GANs within seeds and clean victims."""
    all_results: list[pd.DataFrame] = []
    root = Path(config["experiment"].get("output_dir", "outputs"))
    for dataset_name in datasets:
        local_cfg = copy.deepcopy(config)
        local_cfg["experiment"]["dataset"] = dataset_name
        if dataset_name == "synthetic":
            local_cfg["experiment"]["data_path"] = None
        elif data_path:
            candidate = Path(data_path) / dataset_name
            if candidate.is_dir():
                local_cfg["experiment"]["data_path"] = str(candidate)
            elif len(datasets) == 1:
                local_cfg["experiment"]["data_path"] = data_path
            else:
                raise ValueError(f"Multiple datasets require subdirectories under {data_path}; missing {candidate}")
        elif dataset_name != config["experiment"].get("dataset"):
            local_cfg["experiment"]["data_path"] = str(Path("data/raw") / dataset_name)
        local_cfg["experiment"]["output_dir"] = str(root / dataset_name)
        runner = ExperimentSession(local_cfg)
        for seed in seeds:
            for rate in rates:
                print(f"[experiment] dataset={dataset_name} seed={seed} rate={rate} victims={victims}", flush=True)
                all_results.append(runner.run(seed=seed, rate=rate, victim=victims))
    combined = pd.concat(all_results, ignore_index=True)
    root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(root/"all_results.csv", index=False)
    summary = summarize(combined)
    summary.to_csv(root/"summary_mean_sd_ci95.csv", index=False)
    tests = paired_tests(combined)
    tests.to_csv(root/"paired_wilcoxon_holm.csv", index=False)
    plot_sweep(combined, root/"figures")
    return combined
