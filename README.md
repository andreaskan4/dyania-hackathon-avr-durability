# Dyania Docathlon 2026 — Bioprosthetic Aortic Valve Durability

A clinical decision-support prototype for **aortic valve replacement (AVR) durability**: how likely is a
bioprosthetic valve to fail over time, and — for a proposed procedure — which candidate valve is estimated
to remain event-free the longest for a specific patient?

Built from a small deidentified clinical cohort (labs, medications, clinical notes) and two survival
models trained on it, presented through an interactive Streamlit dashboard.

## Problem Statement

Bioprosthetic aortic valves degrade over time (Structural Valve Deterioration), and a surgeon or
interventional cardiologist choosing a replacement valve today has to weigh several candidate options —
SAVR vs TAVR, different valve models and sizes — largely against **population-level** durability figures,
not figures personalized to the patient in front of them.

The harder problem is that for any real patient, we only ever observe the outcome of the *one* valve they
actually received. We never see what would have happened with a different valve in the same patient, so
directly learning "which valve is best for this patient" from historical data alone is not possible without
some form of counterfactual reasoning.

On top of that, our available cohort is small (117 patients, only 17 with detailed lab/medication
histories) — far too small to train a deep model to reliably predict long-term, patient-specific outcomes
by itself.

## Our Approach

Two complementary survival models, each answering a different piece of the problem above:

| | v1 — Keras | v2 — PyTorch (personalized) |
|---|---|---|
| Question | Will this patient's valve fail, and when? | For this patient, how long will *this specific candidate valve* last, compared to other candidates? |
| Inputs | Patient labs + medications over time | Patient labs + medications + prior valve history (extracted from notes) **+ a proposed candidate valve** |
| Output | Yearly failure hazard → survival curve | Yearly event-free hazard → survival curve, per candidate valve |
| Artifacts | `model/` | `ml/Model v2/` |
| Training notebooks | `ml/Encoding Files/` | `notebooks/` |

To get around the small-cohort and no-counterfactual-outcomes problems, both models are **discrete-time
survival models** (GRU-based) trained on synthetic cohorts that are *grounded* in real patient data — real
physiology, medications, missingness patterns, and (for v2) real prior valve histories — but with
long-term event timing simulated, and, for v2, simulated counterfactual outcomes under valves a patient
never actually received. This lets the v2 model learn to rank candidate valves for the same patient, which
is not learnable from real data alone.

The result is presented through an interactive Streamlit dashboard: cohort exploration, per-patient
drill-down, both models' architecture and metrics, and a live valve-candidate comparison tool with
explainability. Full modeling details in `ml/approach.md`; what the reported metrics do and don't prove is
in `ml/Model v2/VALIDATION_CAVEAT.md`.

## Repository structure

- `data/` — the clinical data: raw deidentified exports, cleaned/flagged versions, and the reprocessed
  real-patient data used by the v2 model. See `data/data_plan.md`.
- `notebooks/` — the v2 model pipeline, in order: note-based valve-history extraction (Qwen LLM) → merge →
  synthetic cohort generation → training → interpretability/demo.
- `protocol/` — exploratory data analysis on the real and synthetic data. See `protocol/study_protocol.md`.
- `ml/` — model artifacts and methodology for both models. See `ml/approach.md`.
- `presentation/demo/` — the Streamlit dashboard (see below); `presentation/slides.pdf` — the pitch deck.

## Running the dashboard

```bash
pip install -r presentation/demo/requirements.txt
streamlit run presentation/demo/app.py
```

Pages: cohort overview; labs/medications/notes exploration; a per-patient explorer; a simple risk-estimator
demo; the v1 Keras model (architecture + metrics); and the v2 personalized valve-comparison tool
(architecture, live predictions, SAVR vs TAVR, explainability).

## Key Design Decisions

- **Semi-synthetic training, not the raw cohort.** With only 17–117 real patients, neither model could
  learn long-term durability directly, so both train on a larger synthetic cohort bootstrapped from real
  trajectories. Real data grounds the inputs; long-term outcomes are simulated. See `ml/approach.md`.
- **Two separate models for two separate questions.** A population-level risk model (v1) answers "will
  this valve fail"; a patient-*and*-valve-specific model (v2) answers "which candidate valve lasts
  longest for this patient" — the second needed an architecture that explicitly compares patient and
  valve representations, not just a bigger version of the first.
- **Missingness is kept, not imputed away.** Both models see an explicit "this value was/wasn't measured"
  signal alongside each lab/medication feature, since absence of a test is itself informative.
- **Clinical notes are mined with an LLM, not regex.** Prior valve procedures (SAVR/TAVR, model, size,
  redo, valve-in-valve) are extracted from free-text notes with Qwen 2.5 7B Instruct
  (`notebooks/01_qwen_valve_extraction.ipynb`), which is far more robust than keyword matching.

## Status

Hackathon proof-of-concept. **Not clinically validated — not for treatment decisions.** See
`ml/Model v2/VALIDATION_CAVEAT.md`.

## Team

**Brute Force**

- Andreas Kandilas — Software Engineer
- Panagiwths Xhovalin Qazimi — ML Engineer
- Ioulios Konstantelos — ML Engineer / Physician
