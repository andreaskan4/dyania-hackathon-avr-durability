# Study Protocol

## Objective

Explore whether a patient's longitudinal clinical record (labs, medications, clinical notes) can support
two related predictions:

1. **Risk:** how likely is a bioprosthetic aortic valve to fail (Structural Valve Deterioration) over
   time? (v1 model)
2. **Choice:** among several candidate replacement valves, which is estimated to remain event-free
   longest for *this* patient? (v2 model)

## Cohort

A single deidentified clinical cohort of aortic valve replacement patients:

- 117 patients with clinical notes and a yearly `Valve_Failure` outcome label.
- 17 of those patients additionally have detailed lab and medication histories, and are the ones used for
  exploratory data analysis and to seed both models' synthetic training cohorts.

Full breakdown and known limitations (small n; lab/medication data only for the 17-patient subset) are in
`data/data_plan.md`.

## Exploratory data analysis

`protocol/00_cleanup_eda_real_and_synthetic.ipynb` covers cleaning and exploring both the real cohort and
the synthetic cohorts derived from it — distributions, missingness, and sanity checks on the
bootstrapped/synthetic data before it is used for model training.

The Streamlit dashboard (`presentation/demo/`) also surfaces this cohort-level EDA interactively: lab,
medication, and note statistics, missingness, correlations, and outcome distributions, without needing to
re-run notebooks.

## Modeling approach

Summarized in `ml/approach.md`. In short: because the real cohort is too small to learn long-term
durability directly, both models train on synthetic cohorts grounded in the real cohort's patterns (real
physiology, medications, missingness, prior valve history) but with simulated long-term event timing.

## Validation

Both models report held-out metrics on their respective synthetic test splits — v1's are in
`ml/Encoding Files/04_survival_evaluation_tensorflow.ipynb`; v2's are in `ml/Model v2/metrics.json` and
`ml/Model v2/baseline_vs_final_ablation.csv`. Because the synthetic cohorts are bootstrapped from a
shared, small pool of real donors reused across splits, these metrics measure recovery of the synthetic
task, **not** generalization to unseen real patients. Full caveat in `ml/Model v2/VALIDATION_CAVEAT.md`.

## Deliverable

An interactive dashboard (`presentation/demo/app.py`) presenting the cohort data, both models, and their
explainability tooling, for demonstration purposes only — not a clinical decision aid.
