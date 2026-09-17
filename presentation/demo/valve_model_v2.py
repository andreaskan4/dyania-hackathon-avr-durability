"""
Personalized valve-durability model (v2) — inference + interpretability.

This is a direct port of the frozen inference pipeline documented in
`notebooks/05_interpretability_and_demo.ipynb`, reorganized as importable
functions instead of a linear notebook so the Streamlit app can run it for
any of the 17 real donor patients and any candidate valve, not just the one
hardcoded demo patient.

Runtime artifacts (model checkpoint, preprocessing.json, encoder spec) live
in `ml/Model v2/`; the preprocessed real donor data lives in
`data/Real Patient Data (v2)/`.

Nothing about the model weights, preprocessing constants, or math has been
changed from the frozen checkpoint / `preprocessing.json` contract.
"""

from __future__ import annotations

import ast
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Frozen architecture (must match the checkpoint's state_dict exactly)
# --------------------------------------------------------------------------- #

class SequenceGRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.proj(h[-1])


class CandidateValveEncoder(nn.Module):
    def __init__(self, n_procedure, n_model, n_position, numeric_dim, output_dim=48):
        super().__init__()
        self.procedure_emb = nn.Embedding(n_procedure, 4)
        self.model_emb = nn.Embedding(n_model, 12)
        self.position_emb = nn.Embedding(n_position, 3)
        in_dim = 4 + 12 + 3 + numeric_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, procedure, model, position, numeric):
        x = torch.cat(
            [self.procedure_emb(procedure), self.model_emb(model), self.position_emb(position), numeric],
            dim=1,
        )
        return self.mlp(x)


class InteractionAwareValveSurvivalModel(nn.Module):
    def __init__(self, lab_input_dim, med_input_dim, history_input_dim, candidate_num_input_dim,
                 candidate_vocabs, n_bins):
        super().__init__()
        self.lab_encoder = SequenceGRUEncoder(input_dim=lab_input_dim, hidden_dim=64, output_dim=32)
        self.med_encoder = SequenceGRUEncoder(input_dim=med_input_dim, hidden_dim=48, output_dim=24)
        self.history_encoder = nn.Sequential(
            nn.Linear(history_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 20),
            nn.ReLU(),
            nn.LayerNorm(20),
        )
        self.patient_projection = nn.Sequential(
            nn.Linear(32 + 24 + 20, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.LayerNorm(48),
        )
        self.candidate_encoder = CandidateValveEncoder(
            n_procedure=len(candidate_vocabs["candidate_procedure_type"]),
            n_model=len(candidate_vocabs["candidate_valve_model"]),
            n_position=len(candidate_vocabs["candidate_valve_position"]),
            numeric_dim=candidate_num_input_dim,
            output_dim=48,
        )
        self.interaction_fusion = nn.Sequential(
            nn.Linear(48 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
        )
        self.survival_head = nn.Linear(64, n_bins)

    def forward(self, batch):
        z_lab = self.lab_encoder(batch["lab"], batch["lab_lengths"])
        z_med = self.med_encoder(batch["med"], batch["med_lengths"])
        z_history = self.history_encoder(batch["history"])
        z_patient = self.patient_projection(torch.cat([z_lab, z_med, z_history], dim=1))
        z_valve = self.candidate_encoder(
            batch["candidate_procedure"], batch["candidate_model"],
            batch["candidate_position"], batch["candidate_numeric"],
        )
        z_product = z_patient * z_valve
        z_absdiff = torch.abs(z_patient - z_valve)
        interaction = torch.cat([z_patient, z_valve, z_product, z_absdiff], dim=1)
        fused = self.interaction_fusion(interaction)
        return self.survival_head(fused)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_artifacts(model_dir: Path, data_dir: Path) -> dict:
    checkpoint = torch.load(model_dir / "personalized_valve_survival_model.pt", map_location=DEVICE, weights_only=False)
    with open(model_dir / "preprocessing.json") as f:
        preproc = json.load(f)

    lab_cols = list(checkpoint["lab_features"])
    med_cols = list(checkpoint["medication_features"])
    history_cols = list(checkpoint["history_features"])
    candidate_num_cols = list(checkpoint["candidate_num_features"])
    candidate_vocabs = checkpoint["candidate_vocabs"]
    n_bins = int(checkpoint["n_bins"])
    bin_width_years = float(checkpoint["bin_width_years"])

    lab_mean = pd.Series(preproc["lab_mean"]).reindex(lab_cols).astype(float)
    lab_std = pd.Series(preproc["lab_std"]).reindex(lab_cols).astype(float)
    history_mean = pd.Series(preproc["history_mean"]).reindex(history_cols).astype(float)
    history_std = pd.Series(preproc["history_std"]).reindex(history_cols).astype(float)
    candidate_num_mean = pd.Series(preproc["candidate_num_mean"]).reindex(candidate_num_cols).astype(float)
    candidate_num_std = pd.Series(preproc["candidate_num_std"]).reindex(candidate_num_cols).astype(float)

    lab_input_dim = len(lab_cols) * 2 + 1
    med_input_dim = len(med_cols) * 2 + 1
    history_input_dim = len(history_cols) * 2
    candidate_num_input_dim = len(candidate_num_cols) * 2

    model = InteractionAwareValveSurvivalModel(
        lab_input_dim, med_input_dim, history_input_dim, candidate_num_input_dim,
        candidate_vocabs, n_bins,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    real_timeline = pd.read_csv(data_dir / "real_17_patient_year_lab_med_with_qwen_anchor.csv")
    real_timeline["Year"] = pd.to_numeric(real_timeline["Year"], errors="coerce")
    qwen_history = pd.read_pickle(data_dir / "qwen_prior_valve_history_17.pkl")
    candidate_catalog = pd.read_csv(data_dir / "observed_candidate_catalog.csv")
    candidate_catalog["candidate_valve_model"] = candidate_catalog["candidate_valve_model"].apply(clean_model_name)

    real_patients = sorted(set(real_timeline["Patient"].dropna()) & set(qwen_history["Patient"].dropna()))

    with open(model_dir / "metrics.json") as f:
        metrics = json.load(f)

    return {
        "model": model,
        "lab_cols": lab_cols,
        "med_cols": med_cols,
        "history_cols": history_cols,
        "candidate_num_cols": candidate_num_cols,
        "candidate_vocabs": candidate_vocabs,
        "n_bins": n_bins,
        "bin_width_years": bin_width_years,
        "horizon_years": n_bins * bin_width_years,
        "lab_mean": lab_mean,
        "lab_std": lab_std,
        "history_mean": history_mean,
        "history_std": history_std,
        "candidate_num_mean": candidate_num_mean,
        "candidate_num_std": candidate_num_std,
        "real_timeline": real_timeline,
        "qwen_history": qwen_history,
        "candidate_catalog": candidate_catalog,
        "real_patients": real_patients,
        "metrics": metrics,
    }


# --------------------------------------------------------------------------- #
# Encoding helpers
# --------------------------------------------------------------------------- #

def clean_model_name(x):
    if pd.isna(x):
        return "UNKNOWN_MODEL"
    x = str(x).strip()
    if x == "" or x.lower() in {"name", "nan", "none", "unknown"}:
        return "UNKNOWN_MODEL"
    return x


def encode_category(value, vocab):
    if pd.isna(value):
        return 0
    value = str(value).strip()
    if value == "" or value.lower() in {"nan", "none"}:
        return 0
    return int(vocab.get(value, 0))


def listify(x):
    if isinstance(x, list):
        return x
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    try:
        y = ast.literal_eval(str(x))
        return y if isinstance(y, list) else []
    except Exception:
        return []


def as_bool(x):
    if x is True:
        return True
    if x is False or x is None:
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def transform_history_vector(art: dict, history_dict: dict) -> torch.Tensor:
    raw = pd.Series({c: history_dict.get(c, np.nan) for c in art["history_cols"]})
    raw = pd.to_numeric(raw, errors="coerce")
    mask = raw.notna().astype(np.float32)
    z = ((raw - art["history_mean"]) / art["history_std"]).fillna(0.0).astype(np.float32)
    x = np.concatenate([z.to_numpy(), mask.to_numpy()]).astype(np.float32)
    return torch.tensor(x, dtype=torch.float32)


def transform_candidate_row(art: dict, candidate_row) -> dict:
    procedure = encode_category(candidate_row["candidate_procedure_type"], art["candidate_vocabs"]["candidate_procedure_type"])
    model_idx = encode_category(clean_model_name(candidate_row["candidate_valve_model"]), art["candidate_vocabs"]["candidate_valve_model"])
    position = encode_category(candidate_row["candidate_valve_position"], art["candidate_vocabs"]["candidate_valve_position"])

    raw_num = pd.Series({c: candidate_row.get(c, np.nan) for c in art["candidate_num_cols"]})
    raw_num = pd.to_numeric(raw_num, errors="coerce")
    mask = raw_num.notna().astype(np.float32)
    z = ((raw_num - art["candidate_num_mean"]) / art["candidate_num_std"]).fillna(0.0).astype(np.float32)
    numeric = np.concatenate([z.to_numpy(), mask.to_numpy()]).astype(np.float32)

    return {"procedure": procedure, "model": model_idx, "position": position,
            "numeric": torch.tensor(numeric, dtype=torch.float32)}


def prepare_real_patient_trajectory(art: dict, patient_id: str) -> dict:
    g = (
        art["real_timeline"][art["real_timeline"]["Patient"].eq(patient_id)]
        .dropna(subset=["Year"])
        .sort_values("Year")
        .copy()
    )
    if len(g) == 0:
        raise ValueError(f"No real trajectory for {patient_id}")

    surgery_year = float(g["Year"].max() + 1)
    time_months = (g["Year"].astype(float) - surgery_year) * 12.0

    lab_raw = g[art["lab_cols"]].apply(pd.to_numeric, errors="coerce")
    lab_mask = lab_raw.notna().astype(np.float32)
    lab_z = ((lab_raw - art["lab_mean"]) / art["lab_std"]).fillna(0.0).astype(np.float32)

    med_raw = g[art["med_cols"]].apply(pd.to_numeric, errors="coerce")
    med_mask = med_raw.notna().astype(np.float32)
    med_filled = med_raw.fillna(0.0).astype(np.float32)

    time_channel = (time_months.to_numpy().reshape(-1, 1) / 120.0).astype(np.float32)

    lab_x = np.concatenate([lab_z.to_numpy(), lab_mask.to_numpy(), time_channel], axis=1).astype(np.float32)
    med_x = np.concatenate([med_filled.to_numpy(), med_mask.to_numpy(), time_channel], axis=1).astype(np.float32)

    return {
        "patient_id": patient_id,
        "surgery_year": surgery_year,
        "source_table": g,
        "lab_raw": lab_raw,
        "med_raw": med_raw,
        "lab": torch.tensor(lab_x, dtype=torch.float32),
        "med": torch.tensor(med_x, dtype=torch.float32),
    }


def prior_history_from_qwen(art: dict, patient_id: str) -> tuple[dict, dict]:
    rows = art["qwen_history"][art["qwen_history"]["Patient"].eq(patient_id)]
    if len(rows) == 0:
        raise KeyError(f"No Qwen history for {patient_id}")
    r = rows.iloc[0]

    types = [str(x).upper().strip() for x in listify(r.get("procedure_types", []))]
    models = [clean_model_name(x) for x in listify(r.get("valve_models", []))]
    known_models = [x for x in models if x != "UNKNOWN_MODEL"]
    sizes = pd.to_numeric(pd.Series(listify(r.get("valve_sizes_mm", []))), errors="coerce").dropna()

    history = {
        "prior_valve_count": int(sum(t in {"SAVR", "TAVR"} for t in types)),
        "prior_has_SAVR": int("SAVR" in types),
        "prior_has_TAVR": int("TAVR" in types),
        "prior_redo_count": int(sum(as_bool(x) for x in listify(r.get("redo_flags", [])))),
        "prior_ViV_count": int(sum(as_bool(x) for x in listify(r.get("ViV_flags", [])))),
        "prior_latest_valve_size_mm": float(sizes.iloc[-1]) if len(sizes) else np.nan,
        "prior_known_model_count": int(len(known_models)),
    }
    readable = {"procedure_types": types, "valve_models": models, "valve_sizes_mm": sizes.tolist()}
    return history, readable


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #

def hazards_to_survival(art: dict, hazard_logits: torch.Tensor):
    hazards = torch.sigmoid(hazard_logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    survival_start = torch.cat(
        [torch.ones((survival.shape[0], 1), dtype=survival.dtype, device=survival.device), survival[:, :-1]],
        dim=1,
    )
    rmst = survival_start.sum(dim=1) * art["bin_width_years"]
    return hazards, survival, rmst


def median_survival_years(art: dict, survival_vector) -> float:
    survival_vector = np.asarray(survival_vector, dtype=float)
    idx = np.where(survival_vector <= 0.5)[0]
    if len(idx) == 0:
        return float("nan")
    return (int(idx[0]) + 1) * art["bin_width_years"]


def survival_at(art: dict, survival_vector, year: float) -> float:
    idx = int(math.ceil(year / art["bin_width_years"]) - 1)
    idx = min(max(idx, 0), len(survival_vector) - 1)
    return float(survival_vector[idx])


def make_single_batch(art: dict, prepared_patient: dict, history_tensor: torch.Tensor, candidate_row) -> dict:
    candidate = transform_candidate_row(art, candidate_row)
    return {
        "lab": prepared_patient["lab"].unsqueeze(0).to(DEVICE),
        "lab_lengths": torch.tensor([prepared_patient["lab"].shape[0]], dtype=torch.long, device=DEVICE),
        "med": prepared_patient["med"].unsqueeze(0).to(DEVICE),
        "med_lengths": torch.tensor([prepared_patient["med"].shape[0]], dtype=torch.long, device=DEVICE),
        "history": history_tensor.unsqueeze(0).to(DEVICE),
        "candidate_procedure": torch.tensor([candidate["procedure"]], dtype=torch.long, device=DEVICE),
        "candidate_model": torch.tensor([candidate["model"]], dtype=torch.long, device=DEVICE),
        "candidate_position": torch.tensor([candidate["position"]], dtype=torch.long, device=DEVICE),
        "candidate_numeric": candidate["numeric"].unsqueeze(0).to(DEVICE),
    }


def predict_candidate(art: dict, prepared_patient: dict, history_tensor: torch.Tensor, candidate_row) -> dict:
    batch = make_single_batch(art, prepared_patient, history_tensor, candidate_row)
    with torch.no_grad():
        logits = art["model"](batch)
        _, survival, rmst = hazards_to_survival(art, logits)
    survival_np = survival[0].detach().cpu().numpy()
    return {
        "rmst_years": float(rmst[0].detach().cpu()),
        "median_survival_years": median_survival_years(art, survival_np),
        "survival_5y": survival_at(art, survival_np, 5),
        "survival_8y": survival_at(art, survival_np, 8),
        "survival_10y": survival_at(art, survival_np, 10),
        "survival_curve": survival_np,
    }


# --------------------------------------------------------------------------- #
# Interpretability
# --------------------------------------------------------------------------- #

def compute_history_drivers(art: dict, prepared_patient: dict, history_tensor: torch.Tensor, reference_candidate_row) -> tuple[pd.DataFrame, float]:
    baseline_rmst = predict_candidate(art, prepared_patient, history_tensor, reference_candidate_row)["rmst_years"]

    def predict_modified(lab_tensor, med_tensor, hist_tensor):
        modified_patient = {"lab": lab_tensor, "med": med_tensor}
        return predict_candidate(art, modified_patient, hist_tensor, reference_candidate_row)["rmst_years"]

    rows = []
    for j, feature in enumerate(art["lab_cols"]):
        lab_mod = prepared_patient["lab"].clone()
        med_mod = prepared_patient["med"].clone()
        lab_mod[:, j] = 0.0
        rmst_occ = predict_modified(lab_mod, med_mod, history_tensor)
        rows.append({"modality": "lab", "feature": feature.replace("lab__", ""), "effect_years": baseline_rmst - rmst_occ})

    for j, feature in enumerate(art["med_cols"]):
        lab_mod = prepared_patient["lab"].clone()
        med_mod = prepared_patient["med"].clone()
        med_mod[:, j] = 0.0
        rmst_occ = predict_modified(lab_mod, med_mod, history_tensor)
        rows.append({"modality": "medication", "feature": feature.replace("med__", "").replace("__present", ""), "effect_years": baseline_rmst - rmst_occ})

    for j, feature in enumerate(art["history_cols"]):
        h_mod = history_tensor.clone()
        h_mod[j] = 0.0
        rmst_occ = predict_modified(prepared_patient["lab"], prepared_patient["med"], h_mod)
        rows.append({"modality": "prior valve history", "feature": feature, "effect_years": baseline_rmst - rmst_occ})

    driver_df = pd.DataFrame(rows)
    driver_df["absolute_effect_years"] = driver_df["effect_years"].abs()
    driver_df = driver_df.sort_values("absolute_effect_years", ascending=False).reset_index(drop=True)
    return driver_df, baseline_rmst


CANDIDATE_FEATURE_GROUPS = [
    "candidate_procedure_type",
    "candidate_valve_model",
    "candidate_valve_size_mm",
    "candidate_valve_position",
]


def _hybrid_candidate(A, B, replaced_features):
    hybrid = A.copy()
    for feature in replaced_features:
        hybrid[feature] = B[feature]
    return hybrid


def compute_candidate_shapley(art: dict, prepared_patient: dict, history_tensor: torch.Tensor, candidate_A, candidate_B) -> tuple[pd.DataFrame, float, float]:
    coalition_value = {}
    for r in range(len(CANDIDATE_FEATURE_GROUPS) + 1):
        for subset_tuple in combinations(CANDIDATE_FEATURE_GROUPS, r):
            subset = frozenset(subset_tuple)
            hybrid = _hybrid_candidate(candidate_A, candidate_B, subset)
            coalition_value[subset] = predict_candidate(art, prepared_patient, history_tensor, hybrid)["rmst_years"]

    n = len(CANDIDATE_FEATURE_GROUPS)
    rows = []
    for feature in CANDIDATE_FEATURE_GROUPS:
        phi = 0.0
        others = [x for x in CANDIDATE_FEATURE_GROUPS if x != feature]
        for r in range(len(others) + 1):
            for subset_tuple in combinations(others, r):
                S = frozenset(subset_tuple)
                S_with = S | {feature}
                weight = math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n)
                marginal = coalition_value[S_with] - coalition_value[S]
                phi += weight * marginal
        rows.append({
            "candidate_feature": feature.replace("candidate_", ""),
            "shapley_delta_years": phi,
            "A_value": candidate_A[feature],
            "B_value": candidate_B[feature],
        })

    shapley_df = pd.DataFrame(rows)
    rmst_A = coalition_value[frozenset()]
    rmst_B = coalition_value[frozenset(CANDIDATE_FEATURE_GROUPS)]
    return shapley_df, rmst_A, rmst_B
