# UI TEAM — START HERE

This ZIP contains the **final notebooks and a ready-to-use UI/runtime bundle**.

You should NOT retrain the model or rerun the full preprocessing pipeline.

## Fastest option: no model execution at all

Build the UI directly from:

- `04_PRECOMPUTED_DEMO/UI_MOCK_RESPONSE.json`
- `04_PRECOMPUTED_DEMO/Patient_101_candidate_comparison.csv`
- `04_PRECOMPUTED_DEMO/Patient_101_history_drivers.csv`
- `04_PRECOMPUTED_DEMO/Patient_101_OBS_006_vs_OBS_013_valve_shapley.csv`
- `04_PRECOMPUTED_DEMO/figures/`

The precomputed demo contains **all 16 observed candidate configurations**, including SAVR and TAVR.

## Current demo patient

**Patient_101**

Highest model-estimated durability in the precomputed demo:

- Candidate: **OBS_013**
- Procedure: **TAVR**
- Model: **Edwards-Sapien**
- Size: **29.0 mm**
- 12-year RMST: **10.75 years**
- 5-year event-free: **98.3%**
- 8-year event-free: **92.3%**
- 10-year event-free: **62.8%**

## If you want live inference in the UI

You still do NOT need to rerun training.

Use the frozen model:

`02_MODEL_RUNTIME/`
- `personalized_valve_survival_model.pt`
- `preprocessing.json`

Use the already-preprocessed real inputs:

`03_PREPROCESSED_REAL_DATA/`
- `real_17_patient_year_lab_med_with_qwen_anchor.csv`
- `qwen_prior_valve_history_17.pkl`
- `observed_candidate_catalog.csv`

The exact inference implementation already exists in:

`01_FINAL_NOTEBOOKS/05_interpretability_and_demo.ipynb`

That notebook can be used as the backend reference.

## Final notebooks

`01_FINAL_NOTEBOOKS/`

- `01_qwen_valve_extraction.ipynb`
- `02_merge_real_multimodal_data.ipynb`
- `03_generate_semisynthetic_training_data.ipynb`
- `04_train_personalized_survival_model.ipynb`
- `05_interpretability_and_demo.ipynb`
- `PROJECT_BRIEF_FOR_PPT_AND_UI.md`

For UI work, notebook **05** is the one that matters most.
The earlier notebooks document how the frozen artifacts were produced.

## Final model metrics

Held-out synthetic-instance evaluation (shared 17-patient real donor library):

- C-index: **0.863**
- within-patient Spearman: **0.823**
- top-1 candidate recovery: **76.0%**
- pairwise candidate-ranking accuracy: **88.3%**

These are not clinical performance estimates.


## Important validation caveat

The train/validation/test split is correctly grouped by **synthetic patient identity**, so the four candidate-valve scenarios for a synthetic patient never cross splits.

However, the 1,000 synthetic patients are bootstrapped from only **17 real donor trajectories**, and those donor identities are reused across train, validation, and test. Therefore the reported metrics measure performance on new synthetic instances from the same donor library. They are **not** donor-independent validation and must not be presented as performance on unseen real patients.

This does not invalidate the hackathon proof-of-concept, but the distinction should be stated clearly in the PPT.

## Recommended UI wording

Use:
- `Model-estimated event-free durability`
- `Candidate comparison`
- `Prediction drivers`
- `Highest model-estimated durability`

Avoid:
- `best valve`
- `recommended treatment`
- `guaranteed lifespan`
- `AI-selected valve`

## Mandatory footer

**Research prototype — data-grounded semi-synthetic proof of concept. Predictions are not clinically validated and should not be used for treatment decisions.**
