# Scientific and Methodological Implementation Notes

## Source

The complete code was created from the user's uploaded 22-page **GAN-Based Data Poisoning Attack and Defence Strategies in Cybersecurity** document (especially Sections 3.3–3.6 and 4.1–4.7). The uploaded document ends after the dataset feature table in Section 5.2; later results and proofs discussed elsewhere in the text were not present in the supplied file. Reported experimental numbers are NOT reused as fresh computations.

## Equation (7), poison labels, and surrogate

The manuscript defines the generator's two competing goals (adversarial realism and victim-classifier degradation), with a λ trade-off. It does not define a fully differentiable outer optimization through repeated victim retraining, nor the exact labels given to generated samples. The implementation therefore uses an explicitly **single-level differentiable surrogate approximation**:

```
L_G = BCEWithLogits(D_attacker(G(z)), real=1)
      - lambda * CE(surrogate_attacker(G(z)), injected_label)
```

The attacker GAN is trained to resemble the most common non-normal attack class in the separate auxiliary partition. In the default dirty-label mode, those attack-like generated records are **injected with the benign label**; this creates label conflicts. The surrogate cross-entropy is maximized for that injected benign label. In experimental clean-label mode, the generated records keep their source class. The latter is a defined experimental variant, not a strong claim to reproduce a specific clean-label attack from the manuscript.

This is a training-data poisoning benchmark, not test-time adversarial evasion. The poisoning budget `epsilon` is a fraction of the clean victim-training count. The observed poisoned fraction among *all combined* records is `M/(N+M)` and is separately recorded.

## Stage 1 calibration and scalability

The exact Stage 1 uses `sklearn.cluster.DBSCAN`, Euclidean distances, and a trusted-set 95th-percentile minPts-th neighbor radius. This is NOT a formal per-class guarantee of 95% clean retention after fitting DBSCAN to the contaminated mixture. Exact DBSCAN's worst-case memory and time can grow quadratically. The `reference_knn` fallback computes trusted-neighbor distances for all candidates. It is **a scalable approximation**, not exact DBSCAN. The mode used appears in the audit and run metadata. Full-data claims must explicitly disclose which mode was used.

## Stage 2 calibration

The defender's GAN and discriminator train solely on its 80% trusted-training partition. Its remaining 20% trusted records calibrate Stage 2 scores; using the empirical order statistic at index `floor(0.05*n)` makes rejection on that calibration set at most 5% (ties may reduce the observed fraction). The manuscript's 5% target and description of a '20% smallest score' were internally inconsistent; the 5% criterion was selected as the reproducible operational rule. Overall clean retention on the contaminated candidate set is independently reported, without assuming that 5% at each stage means 5% in total.

## Defender independence

- Attacker knows training feature schema and its own auxiliary training records only.
- Victim clean train records, trusted defender train/calibration records, and untouched test data use disjoint row IDs.
- The two GANs are initialized separately (`attacker seed`, `defender seed + 100`), optimized independently, and train on disjoint subsets.
- The deployed `train_defender()` and `apply_defense()` functions do **not** accept attacker GANs, attack-source labels, `is_poison` flags, or injected-record ground truth.
- The attacker discriminator is only accessible to an explicitly named optional `attacker_discriminator_oracle`, disabled by default. This cannot be used for a deployed defender claim.

## Dataset handling and feature counts

Data cleaning is performed prior to splitting to prevent exact duplicate leakages; the model feature vocabulary and MinMax parameters are fitted using the **80% clean training partition only**, never validation/test rows. The 80% training data are subsequently split equally into attacker and victim partitions. The complete 10% validation partition is defender-trusted, and its 80/20 subdivision is not shared with the attacker.

The manuscript reports 71 final CIC-IDS2017 input features (68 numeric + protocol one-hot) and 76 final TON_IoT network input features (22 numeric/Boolean + 54 categorical one-hot). Actual counts depend on CSV schema, dropped constants, train-only categories, and release version. They are explicitly recorded and compared in `dataset_manifest.json`, never manufactured to match the expected number. TON_IoT flags with T/F/- representations are encoded as binary when recognized. The network-only subset should be used for TON_IoT.

UNSW-NB15 can be a separate validation dataset. The code re-splits provided official train/test CSV files into 80/10/10 to match the manuscript protocol. This differs from experiments that preserve the official train/test partition; compare with care.

## Victim architectures

The paper names Random Forest, XGBoost and 1D CNN but does not give all tree hyperparameters or the complete CNN layer layout. The implementation uses fixed, documented hyperparameters in YAML and a two-convolution CNN: Conv1D(1→32,k=3)→ReLU→Conv1D(32→64,k=3)→ReLU→AdaptiveAvgPool→Dense(64→64)→Dense(n_classes). It is implemented in PyTorch rather than the manuscript's Keras model. Thus the paper's CNN numbers cannot be assumed to reproduce exactly.

## Baseline interpretation

- **Spectral signatures:** class-conditional top singular direction of centered raw features. The original learned representation for spectral-signatures literature is not specified in the manuscript. The repository names the method but does not imply architecture-equivalent prior-work reproduction.
- **Gradient influence proxy:** approximate per-example training-loss gradient conflict with held-out trusted loss gradient at a logistic reference model; **not** exact Hessian-inverse influence functions.
- **Deep k-NN:** trained neural representation, followed by trusted neighbor label-consistency and distance scoring.
- **FGSM adversarial training:** CNN-only numerical-feature-space FGSM with one-hot groups unchanged; it does **not** implement a physically constrained packet generator. Traditional tree models do not support this direct input-gradient procedure; no proxy is mislabeled FGSM.
- **Oracle:** uses attacker-side discriminator and must only be interpreted as a non-deployable, attacker-informed diagnostic.

Baseline data-removal budgets are set from **known nominal attack injection count**, not known poison sample IDs; this is a budget-aware benchmark setting and may favor baselines compared to an agnostic deployment setting. The full dual-layer defence makes its decisions without poison IDs or target labels.

## Statistical practice

Five-seed sweeps compute sample mean, sample SD (`ddof=1`), Student-t 95% CIs, and paired Wilcoxon with Holm adjustment and paired effect sizes *within each dataset/model/budget comparison family*. Statistical tests are only produced for at least five matched seeds, and one-seed examples correctly have no CI. A single-model one-seed result is not a statistically supported performance claim.

## Performance caution

The synthetic demo is a **functionality test**, not a published-data benchmark. The paper's reported 65.2% / 83.9% / 85.4% values are not included as experimental outputs and cannot be assumed achievable. Datasets are not bundled, and no results are invented. Real datasets may have schema/version, class imbalance, runtime, or missing-class constraints requiring a careful study-specific configuration.
