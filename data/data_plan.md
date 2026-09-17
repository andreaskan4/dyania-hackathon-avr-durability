# Data Plan

## Source

One small deidentified clinical cohort of aortic valve replacement (AVR) patients — labs, medications,
and clinical notes — exported as `Deidentified Dataset/*.xlsx`.

Of the patients in this cohort:

- **117 patients** have clinical notes and a per-year outcome label (`Valve_Failure`).
- **17 patients** additionally have detailed lab and medication histories.

Because 17–117 real patients is too small to train a reliable deep model, both models in this project
train on a *larger synthetic cohort* built by bootstrapping and perturbing these real trajectories — not
on the real cohort directly. The real data is used to (a) ground the synthetic generators in realistic
values and patterns, and (b) run live, patient-specific demos in the dashboard. See `ml/approach.md`.

## Folders

| Folder | Contents | Used by |
|---|---|---|
| `Deidentified Dataset/` | Raw deidentified exports (`.xlsx`): labs, medications, notes | source of everything below |
| `Clean Synthesised Datasets/` | Cleaned & flagged CSVs: labs, medications, notes, and a rolled-up patient-year table with the `Valve_Failure` label | dashboard (Labs / Medications / Clinical Notes / Multimodal Cohort pages), v1 Keras model |
| `Real Patient Data (v2)/` | The 17 richly-documented patients, reprocessed for the v2 model: a patient-year lab/medication timeline, LLM-extracted prior valve history, and a catalog of 16 observed candidate valve configurations | v2 model, dashboard (Valve Candidate Comparison page) |

## Preprocessing summary

- Dates are shifted/reduced to year-only granularity for deidentification.
- Labs and medications are pivoted from long (one row per result/order) to a per-patient-year table, with
  numeric aggregation for labs and presence flags per drug class for medications.
- Missing lab/medication values are **kept as missing**, not imputed away — both models receive an
  explicit missingness flag alongside each value, since absence of a test is itself informative.
- Prior valve procedures (SAVR/TAVR, model, size, redo, valve-in-valve) are extracted from free-text
  clinical notes with a Qwen 2.5 7B Instruct LLM (`notebooks/01_qwen_valve_extraction.ipynb`), producing
  `Real Patient Data (v2)/qwen_prior_valve_history_17.pkl`.

## Known limitation

Only 17 of 117 patients have detailed lab/medication data, so lab-based statistics and both models lean
heavily on that subset. This is surfaced directly in the dashboard — e.g. the missingness chart on the
"Multimodal Cohort" page.
