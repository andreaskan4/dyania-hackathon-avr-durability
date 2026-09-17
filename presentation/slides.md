# Presentation Slide Outline

> **Instructions:** Use this as a guide to build your `slides.pdf` (5–10 slides). Export to PDF and place it in this folder as `slides.pdf`. The outline below maps to the evaluation rubric.

---

## Slide 1 — Title

- Team name
- Challenge title
- Team members and roles
- Date

---

## Slides 2–4 — Problem Framing

**Slide 2: The Clinical Problem**
- What is structural valve deterioration (SVD) and why does long-term durability matter for AVR patients?
- How does the SAVR vs. TAVR durability debate compound the uncertainty?
- Quantify the gap: how late are failing valves currently identified relative to the ideal reintervention window?

**Slide 3: Current Workflow Failure**
- How is durability risk monitored today (fixed-interval echo surveillance)?
- Where does the bottleneck occur (irregular follow-up, inter-observer variability, delayed reintervention referral)?
- What is the cost of the current approach (clinical, financial, patient outcomes)?

**Slide 4: Our Hypothesis**
- If we could stratify patients by durability risk earlier using routinely collected echo/registry data, what changes?
- What is the surveillance optimisation opportunity?

---

## Slides 5–7 — Proposed Study Design

**Slide 5: Study Overview**
- Target population (inclusion / exclusion criteria — summary)
- Primary and secondary endpoints
- Ground truth definition for structural valve deterioration

**Slide 6: Data Sources**
- What data sources does your system use?
- How realistic is access to these data in a clinical deployment?
- What did you use for the prototype?

**Slide 7: Validation Plan**
- Train / test strategy, handling of censored follow-up
- Key evaluation metrics and why they matter clinically
- Comparator baseline

---

## Slides 8–9 — ML Approach

**Slide 8: Model Architecture**
- What model(s) did you choose and why (survival model, classifier at fixed horizon, sequence model)?
- Feature engineering highlights
- How does the model output map to a clinical action (surveillance interval, reintervention referral)?

**Slide 9: Results / Proof of Concept**
- If you ran the model: key performance numbers (time-dependent AUROC, C-index, calibration)
- SHAP or feature importance summary
- If prototype only: expected performance range and how you would validate it

---

## Slides 10 — Impact and Next Steps

- What does this system enable that isn't possible today?
- What would validation in a real health system / valve registry require?
- Immediate next step if given 3 more months
- One sentence on real-world deployment readiness
