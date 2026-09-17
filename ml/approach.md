# ML Methodology

Two separate survival models, both discrete-time and GRU-based, trained on synthetic cohorts grounded in
the real data (see `data/data_plan.md`).

## v1 — Cohort-level valve-failure risk (`model/`, `ml/Encoding Files/`)

**Question:** given a patient's history so far, what is their yearly risk of valve failure over the next
10 years?

- **Inputs:** three branches — static valve features, a yearly sequence of lab values, a yearly sequence
  of medication-class presence flags.
- **Architecture:** one GRU per sequence branch (physiology, medications) plus a dense branch for static
  features, concatenated and fused through dense layers into 10 sigmoid outputs — one yearly hazard each,
  for years 1–10.
- **Training data:** a synthetic cohort of 1,000 patients (7,169 patient-year rows) bootstrapped from the
  real cohort.
- **Reported metrics** (held-out synthetic test split): Harrell C-index 0.880, 5-year time-dependent AUC
  0.971, Brier score 0.038 at 5 years.
- **Ablation:** physiology alone reaches most of the discrimination (C-index 0.879); the static valve
  branch alone is close to chance (0.475); the full multimodal model does not clearly beat
  physiology-only — physiology is doing most of the work.
- **Notebooks:** `ml/Encoding Files/04_survival_evaluation_tensorflow.ipynb`,
  `05_ablation_study_tensorflow.ipynb`, `05_shap_analysis_tensorflow.ipynb`.

## v2 — Personalized valve-choice comparison (`ml/Model v2/`, `notebooks/`)

**Question:** for *this* patient, given a *proposed candidate valve* (procedure type, model, size), how
long is that valve estimated to remain event-free — and how does the estimate change if a different valve
is proposed instead?

- **Inputs:** a patient's longitudinal labs and medications (GRU-encoded, same idea as v1), their prior
  valve history extracted from clinical notes by an LLM, **and** the proposed candidate valve's
  characteristics.
- **Architecture:** a patient representation (`z_patient`, from the lab/medication GRUs plus a
  prior-history MLP) and a valve representation (`z_valve`, from a candidate-valve encoder) are combined
  through explicit interaction features — `z_patient`, `z_valve`, their elementwise product, and their
  absolute difference — before a discrete-time survival head (12 yearly hazards).
- **Why interaction features:** an earlier baseline without them discriminated overall risk about as well
  (C-index 0.878) but was much worse at the actual product question — *ranking candidate valves correctly
  for the same patient* (pairwise ranking accuracy 68.8% vs 88.3% for the final model). See
  `ml/Model v2/baseline_vs_final_ablation.csv`.
- **Training data:** a semi-synthetic cohort bootstrapped from the same 17 real donor patients, with
  simulated long-term durability and *counterfactual* outcomes under valves the patient never actually
  received — needed because a real patient only ever shows the outcome for the one valve they got.
- **Reported metrics:** C-index 0.863, pairwise candidate-ranking accuracy 88.3%, top-1 candidate recovery
  76%. **Important:** all 17 real donor trajectories are reused across train/validation/test, so these
  measure recovery of the synthetic ranking task within that shared donor library, not generalization to
  unseen real patients — see `ml/Model v2/VALIDATION_CAVEAT.md`.
- **Explainability:** grouped-occlusion sensitivity for patient-history features, and an exact Shapley
  decomposition (over 4 candidate-valve feature groups) explaining *why* the estimate changes between two
  candidate valves.
- **Notebooks, in order:** `01_qwen_valve_extraction` → `02_merge_real_multimodal_data` →
  `03_generate_semisynthetic_training_data` → `04_train_personalized_survival_model` →
  `05_interpretability_and_demo`.

## Common thread

Both models are **proof-of-concept, semi-synthetic**: real physiology, medications, missingness patterns,
and (for v2) prior valve history are grounded in real data, but long-term event timing — and, for v2,
counterfactual outcomes under valves not actually implanted — is simulated. Neither model is clinically
validated.
