# PoisonShieldAI / PoisonGenDefNet

## GAN-Based Data Poisoning Attack and Defense for Network Intrusion Detection

PoisonShieldAI and PoisonGenDefNet provide an integrated framework for simulating data poisoning attacks and evaluating defense strategies in network intrusion detection systems.

The framework combines Generative Adversarial Networks (GANs), machine learning classifiers, deep learning models, data sanitization techniques, and explainable evaluation metrics to analyze the impact of data poisoning attacks and improve model robustness.

## Project Overview

The project implements a classifier-aware GAN-based poisoning framework and a two-stage defense mechanism to identify and remove malicious training samples.

### Key Features

- Network intrusion detection using machine learning and deep learning.
- GAN-based data poisoning attack simulation.
- Classifier-aware poisoning using an attack-surrogate model.
- Two-stage data sanitization and defense mechanism.
- Independent GAN-based discriminator for detecting poisoned samples.
- DBSCAN-based anomaly detection.
- Multiple victim classification models.
- Performance comparison between different defense techniques.
- Detailed evaluation metrics and statistical analysis.
- Automated experiment execution through a command-line interface.
- Visualization of classification and defense performance.

## Objectives

1. Develop a GAN-based data poisoning attack framework for generating adversarial training samples.
2. Implement a two-stage defense mechanism using DBSCAN and an independent GAN discriminator.
3. Evaluate the effectiveness of different defense strategies using machine learning and deep learning classifiers.
4. Analyze model performance using classification metrics, statistical tests, and visualization techniques.

## System Architecture

The framework consists of the following major components:

1. Dataset Preparation
2. Data Preprocessing
3. Attacker Data Partitioning
4. GAN-Based Poisoning Attack
5. Victim Model Training
6. Two-Stage Defense Mechanism
7. Model Retraining
8. Performance Evaluation
9. Visualization and Reporting

## Implemented Modules

| Module | Description |
|---|---|
| Data Preparation | CSV loading, duplicate removal, missing-value handling, categorical encoding, and feature scaling |
| Data Partitioning | Separates attacker-accessible and victim-training datasets |
| PoisonGenDefNet | Three-layer GAN architecture with 128/256/128 hidden units |
| Attack Surrogate | Classifier-aware surrogate model for optimizing poisoning samples |
| Poisoning Budgets | Supports 0.5%, 1%, 5%, 10%, and 20% poisoning rates |
| Victim Models | Random Forest, XGBoost, and PyTorch 1D CNN |
| Defense Stage 1 | DBSCAN-based anomaly detection |
| Defense Stage 2 | Independent GAN discriminator trained using trusted data |
| Ablation Studies | Evaluates DBSCAN-only, discriminator-only, and combined defense |
| Baseline Methods | Spectral signatures, influence-based detection, deep k-NN, and adversarial training |
| Auditing | Records sample-level detection decisions and defense scores |
| Evaluation | Accuracy, precision, recall, F1-score, detection rate, and clean-data retention |
| Visualization | Generates performance comparison plots and evaluation figures |

## Technologies Used

- Python
- PyTorch
- Scikit-learn
- XGBoost
- NumPy
- Pandas
- Matplotlib
- Seaborn
- PyYAML
- Imbalanced-learn
- Pytest

## Installation

### Requirements

- Python 3.10 or later
- Minimum 8 GB RAM recommended
- GPU support optional
- Windows, Linux, or macOS

### Step 1: Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd PoisonShieldAI
```

### Step 2: Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
pip install -e ".[dev]"
```

For GPU acceleration, install the appropriate PyTorch version for your CUDA environment.

## Quick Start

Run the default experiment:

```bash
poisonshield run --config configs/quick_experiment.yaml
```

Alternatively:

```bash
python -m poisonshield.cli run --config configs/quick_experiment.yaml
```

This executes the complete workflow, including:

1. Dataset preparation.
2. Feature preprocessing.
3. GAN-based poisoning attack.
4. Victim model training.
5. DBSCAN-based detection.
6. Independent GAN-based defense.
7. Baseline defense comparisons.
8. Model evaluation.
9. Visualization generation.

### Run Tests

```bash
pytest -q
```

### View Available Commands

```bash
python -m poisonshield.cli --help
```

## Datasets

The framework supports the following network intrusion detection datasets.

### 1. CIC-IDS2017

CIC-IDS2017 contains network traffic records representing normal activities and different types of cyberattacks.

**Official Dataset:**

https://www.unb.ca/cic/datasets/ids-2017.html

### 2. TON_IoT

TON_IoT is a cybersecurity dataset containing network traffic and IoT-related information.

**Official Dataset:**

https://research.unsw.edu.au/projects/toniot-datasets

### 3. UNSW-NB15

UNSW-NB15 contains normal network traffic and multiple categories of network attacks.

**Official Dataset:**

https://research.unsw.edu.au/projects/unsw-nb15-dataset

## Dataset Directory Structure

Place the downloaded datasets inside the following directories:

```text
data/
└── raw/
    ├── cicids2017/
    │   ├── Monday-WorkingHours.pcap_ISCX.csv
    │   ├── Tuesday-WorkingHours.pcap_ISCX.csv
    │   └── Other CSV files
    │
    ├── toniot/
    │   └── Train_Test_Network.csv
    │
    └── unswnb15/
        ├── UNSW_NB15_training-set.csv
        └── UNSW_NB15_testing-set.csv
```

### Dataset Preparation Notes

- Download the official CSV versions of the datasets.
- Extract the files into their corresponding directories.
- The TON_IoT loader uses network traffic records.
- The UNSW-NB15 loader supports the official training and testing CSV files.
- Dataset preprocessing and feature dimensions are recorded in the dataset manifest.

## Running Experiments

### 1. CIC-IDS2017 Binary Classification

```bash
poisonshield run \
  --config configs/full_experiments.yaml \
  --dataset cicids2017 \
  --task binary \
  --data-path data/raw/cicids2017 \
  --victim rf \
  --poison-rate 0.10 \
  --seed 0 \
  --output-dir outputs/cicids2017_binary
```

### 2. TON_IoT Multiclass Classification

```bash
poisonshield run \
  --config configs/full_experiments.yaml \
  --dataset toniot \
  --task multiclass \
  --data-path data/raw/toniot \
  --victim all \
  --poison-rate 0.05 \
  --seed 1 \
  --output-dir outputs/toniot
```

### 3. UNSW-NB15 Classification

```bash
poisonshield run \
  --config configs/full_experiments.yaml \
  --dataset unswnb15 \
  --task multiclass \
  --data-path data/raw/unswnb15 \
  --victim cnn \
  --poison-rate 0.01 \
  --seed 2 \
  --output-dir outputs/unswnb15
```

## GAN-Based Data Poisoning Attack

PoisonGenDefNet implements a classifier-aware GAN architecture to generate poisoning samples.

### Main Components

1. Generator: Produces adversarial training samples.
2. Discriminator: Evaluates the generated samples.
3. Attack Surrogate: Guides the generator using classification feedback.
4. Poisoning Injection: Introduces generated samples into the victim training dataset.

### Supported Poisoning Rates

| Poisoning Rate | Description |
|---|---|
| 0.5% | Very low poisoning level |
| 1% | Low poisoning level |
| 5% | Moderate poisoning level |
| 10% | High poisoning level |
| 20% | Very high poisoning level |

The poisoning budget is calculated as the number of injected samples divided by the clean victim-training dataset size.

## Two-Stage Defense Mechanism

PoisonShieldAI implements two independent defense stages.

### Stage 1: DBSCAN-Based Detection

DBSCAN identifies suspicious samples based on their distance and density within the feature space.

Main parameters:

- Distance metric: Euclidean
- Minimum samples: 10
- Epsilon: Estimated using trusted-data nearest-neighbor distances

### Stage 2: Independent GAN-Based Detection

The second stage uses an independently trained GAN discriminator.

Main operations:

1. Train the discriminator using trusted clean data.
2. Evaluate the remaining training samples.
3. Calculate discriminator scores.
4. Apply the calibrated detection threshold.
5. Identify and remove suspicious samples.

### Combined Defense

The complete defense pipeline combines DBSCAN and the independent GAN discriminator to improve poisoning detection.

## Victim Classification Models

The framework supports three classification algorithms.

| Model | Description |
|---|---|
| Random Forest | Ensemble-based machine learning classifier |
| XGBoost | Gradient-boosting classification algorithm |
| 1D CNN | Deep learning model for network traffic feature classification |

The framework evaluates model performance before and after applying the defense mechanism.

## Baseline Defense Methods

The following methods are included for comparison:

1. DBSCAN-only defense.
2. Independent GAN discriminator-only defense.
3. Combined DBSCAN and GAN defense.
4. Spectral signatures.
5. Gradient-based influence proxy.
6. Deep k-Nearest Neighbors.
7. CNN feature-space FGSM adversarial training.

## Complete Five-Seed Experiments

To execute experiments across multiple datasets, models, poisoning rates, and random seeds:

```bash
poisonshield sweep \
  --config configs/full_experiments.yaml \
  --datasets cicids2017 toniot \
  --victim all \
  --rates 0.005 0.01 0.05 0.10 0.20 \
  --seeds 0 1 2 3 4
```

### Custom Attack Configuration

To change the attack optimization parameter:

```bash
poisonshield run \
  --config configs/full_experiments.yaml \
  --attack-lambda 1.0 \
  --poison-rate 0.20 \
  --dataset cicids2017 \
  --data-path data/raw/cicids2017
```

## Configuration

The project provides two main configuration files.

| Configuration File | Purpose |
|---|---|
| `configs/quick_experiment.yaml` | CPU-compatible quick execution |
| `configs/full_experiments.yaml` | Complete experiment configuration |

### Important Parameters

| Parameter | Description |
|---|---|
| `experiment.task` | Binary or multiclass classification |
| `experiment.random_split_seed` | Dataset splitting seed |
| `experiment.seed` | Model and attack random seed |
| `experiment.victim` | Selects RF, XGBoost, CNN, or all models |
| `attack.label_policy` | Controls poisoning label assignment |
| `attack.lambda_weight` | Controls the classification-loss contribution |
| `defense.dbscan_mode` | Selects exact or scalable DBSCAN processing |
| `experiment.include_baselines` | Enables baseline defense comparisons |
| `experiment.include_oracle` | Enables the attacker-informed comparison method |

A dataset-specific configuration can also be supplied using `--dataset-config`.

## Evaluation Metrics

The framework evaluates classification and defense performance using the following metrics.

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1-score
- Confusion Matrix

### Defense Metrics

- Poison Detection Rate
- Clean Sample Retention
- False Positive Rate
- False Negative Rate
- Recovery Performance

### Statistical Analysis

- Five-seed mean and standard deviation.
- 95% Student-t confidence intervals.
- Paired Wilcoxon statistical test.
- Holm multiple-comparison correction.
- Paired effect sizes.

## Output Directory Structure

All experiment outputs are stored inside the `outputs/` directory.

```text
outputs/
└── experiment_name/
    ├── effective_config.json
    ├── dataset_manifest.json
    │
    ├── dataset_task_seed_rate/
    │   ├── attack_defense_details.json
    │   ├── attacker_gan_training.csv
    │   ├── defender_gan_training.csv
    │   ├── evaluation_provenance.csv.gz
    │   ├── sanitization_audit.csv.gz
    │   ├── confusion_matrices.json
    │   ├── metrics.csv
    │   ├── defended_rf.joblib
    │   ├── defended_xgboost.joblib
    │   ├── defended_cnn.pt
    │   │
    │   └── figures/
    │       └── macro_f1.png
    │
    ├── all_results.csv
    ├── summary_mean_sd_ci95.csv
    ├── paired_wilcoxon_holm.csv
    │
    └── figures/
        └── rate_sweep.png
```

The output files contain:

- Experiment configuration.
- Dataset information.
- GAN training information.
- Poisoning and defense details.
- Sample-level sanitization decisions.
- Classification metrics.
- Confusion matrices.
- Trained model files.
- Performance comparison plots.
- Statistical analysis results.

## Project Directory Structure

```text
PoisonShieldAI/
│
├── configs/
│   ├── quick_experiment.yaml
│   └── full_experiments.yaml
│
├── data/
│   └── raw/
│       ├── cicids2017/
│       ├── toniot/
│       └── unswnb15/
│
├── examples/
│
├── notebooks/
│   └── PoisonShieldAI_Quickstart.ipynb
│
├── src/
│   └── poisonshield/
│       ├── data/
│       │   ├── data loading
│       │   ├── preprocessing
│       │   └── splitting
│       │
│       ├── attacks/
│       │   ├── GAN architecture
│       │   └── poisoning injection
│       │
│       ├── defenses/
│       │   ├── DBSCAN detection
│       │   ├── GAN discriminator
│       │   └── baseline defenses
│       │
│       ├── models/
│       │   ├── CNN
│       │   ├── Random Forest
│       │   ├── XGBoost
│       │   └── training utilities
│       │
│       ├── evaluation/
│       │   ├── classification metrics
│       │   ├── defense metrics
│       │   ├── statistical analysis
│       │   └── visualization
│       │
│       ├── pipeline.py
│       └── cli.py
│
├── tests/
│
├── outputs/
│
├── IMPLEMENTATION_NOTES.md
├── LICENSE
├── pyproject.toml
└── README.md
```

## Security and Data Handling

The framework performs poisoning attack simulation and defense evaluation using local datasets and locally trained models.

It does not interact with remote intrusion detection systems, deployed security appliances, or production network infrastructure.

The defense mechanism operates using trusted training records, mixed training features, an independently trained discriminator, and configured detection thresholds.

## Implementation Notes

The following implementation details are important:

- Dataset feature dimensions depend on the input dataset version and preprocessing configuration.
- CIC-IDS2017 and TON_IoT feature counts are checked against the reference dimensions.
- DBSCAN can use an exact or scalable nearest-neighbor-based processing mode.
- The scalable reference-neighbor approach is an approximation and is not identical to exact DBSCAN.
- The defense calibration threshold does not guarantee the same clean-data retention rate across the complete training dataset.
- GAN training results can vary depending on the dataset, model, random seed, and training configuration.

For additional technical details, refer to `IMPLEMENTATION_NOTES.md`.

## License

This project is distributed under the MIT License.

Dataset copyrights, terms of use, and licenses remain with their respective authors and institutions.

## Acknowledgements

We acknowledge the researchers and institutions responsible for providing the CIC-IDS2017, TON_IoT, and UNSW-NB15 datasets, along with the developers of the open-source machine learning and deep learning libraries used in this project.
