"""
Dyania Docathlon 2026 — Bioprosthetic Aortic Valve Durability Dashboard

A Streamlit dashboard over the project's clinical cohort data (labs, medications,
clinical notes, and the patient-year multimodal table) plus a look inside the
trained Keras multimodal survival model (architecture, reported evaluation
metrics, and the modality ablation study).

Run with:
    streamlit run presentation/demo/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "Clean Synthesised Datasets"
MODEL_PATH = ROOT / "model" / "multimodal_valve_survival_model_v1.keras"

LABS_CSV = DATA_DIR / "labs_cleaned_flagged.csv"
MEDS_CSV = DATA_DIR / "medications_cleaned_flagged.csv"
NOTES_CSV = DATA_DIR / "notes_cleaned.csv"
MULTIMODAL_CSV = DATA_DIR / "multimodal_patient_year_cleaned.csv"

V2_MODEL_DIR = ROOT / "ml" / "Model v2"
V2_DATA_DIR = ROOT / "data" / "Real Patient Data (v2)"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import valve_model_v2 as vm  # noqa: E402

# --------------------------------------------------------------------------- #
# Palette (validated categorical / sequential / diverging / status colors)
# --------------------------------------------------------------------------- #

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0.0, "#184f95"], [0.5, "#f0efec"], [1.0, "#d03b3b"]]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

FONT = dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK_SECONDARY)


def style(fig: go.Figure, legend: bool | None = None) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=FONT,
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=AXIS)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    if legend is not None:
        fig.update_layout(showlegend=legend)
    return fig


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner=False)
def load_model_architecture(path: Path) -> dict | None:
    """Parse the .keras (HDF5) file directly so the dashboard never needs a
    TensorFlow install just to describe the model."""
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        model_config = json.loads(f.attrs["model_config"])
        training_config = json.loads(f.attrs["training_config"]) if "training_config" in f.attrs else {}

        def _decode(v):
            return v.decode() if isinstance(v, bytes) else str(v)

        keras_version = _decode(f.attrs.get("keras_version", ""))
        backend = _decode(f.attrs.get("backend", ""))

        param_counts: dict[str, int] = {}

        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                layer_name = name.split("/")[0]
                param_counts[layer_name] = param_counts.get(layer_name, 0) + obj.size

        if "model_weights" in f:
            f["model_weights"].visititems(_visit)

    layers = []
    for l in model_config["config"]["layers"]:
        name, cls, cfg = l["name"], l["class_name"], l["config"]
        bits = []
        if cls == "InputLayer":
            bits.append(f"shape={cfg.get('batch_input_shape')}")
        if cfg.get("units") is not None:
            bits.append(f"units={cfg['units']}")
        if cfg.get("activation"):
            bits.append(f"activation={cfg['activation']}")
        if cls == "Masking":
            bits.append(f"mask_value={cfg.get('mask_value')}")
        if cls == "Dropout":
            bits.append(f"rate={cfg.get('rate')}")
        layers.append(
            {
                "layer": name,
                "type": cls,
                "config": ", ".join(bits),
                "params": int(param_counts.get(name, 0)),
            }
        )

    return {
        "layers": layers,
        "total_params": sum(l["params"] for l in layers),
        "inputs": model_config["config"].get("input_layers", []),
        "outputs": model_config["config"].get("output_layers", []),
        "keras_version": keras_version,
        "backend": backend,
        "training_config": training_config,
    }


@st.cache_resource(show_spinner="Loading valve durability model (v2)...")
def load_v2_artifacts(model_dir: Path, data_dir: Path):
    return vm.load_artifacts(model_dir, data_dir)


# --------------------------------------------------------------------------- #
# Page config + sidebar navigation
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Dyania Docathlon 2026 — AVR Durability Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 300px;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.4rem;
        margin-top: 0.25rem;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 0.6rem 0.8rem;
        border-radius: 0.5rem;
        transition: background-color 0.1s ease-in-out;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: rgba(128, 128, 128, 0.15);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 1.02rem;
        line-height: 1.4;
        margin: 0;
        white-space: normal;
    }
    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 100%;
        padding-left: 3rem;
        padding-right: 3rem;
        transition: max-width 0.2s ease-in-out;
    }
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    button[aria-label="Close sidebar"],
    button[aria-label="Open sidebar"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    "Overview",
    "Labs",
    "Medications",
    "Clinical Notes",
    "Multimodal Cohort",
    "Patient Explorer",
    "Risk Estimator",
    "Keras Model",
    "Valve Candidate Comparison (v2)",
    "Valve Durability Model (v2)",
]

with st.sidebar:
    st.title("AVR Durability")
    st.caption("Dyania Docathlon 2026")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")

missing = [p for p in [LABS_CSV, MEDS_CSV, NOTES_CSV, MULTIMODAL_CSV] if not p.exists()]
if missing:
    st.error("Missing expected data file(s):\n" + "\n".join(f"- `{m}`" for m in missing))
    st.stop()

labs = load_csv(LABS_CSV)
meds = load_csv(MEDS_CSV)
notes = load_csv(NOTES_CSV)
multi = load_csv(MULTIMODAL_CSV)

LAB_COLS = [c for c in multi.columns if c.startswith("lab__")]
MED_FLAG_COLS = [c for c in multi.columns if c.startswith("med__") and c.endswith("__present")]
MENTION_COLS = [c for c in multi.columns if c.startswith("note_mentions_")]
RICH_PATIENTS = set(labs["Patient"].unique()) | set(meds["Patient"].unique())


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #

def page_overview():
    st.title("Bioprosthetic Aortic Valve Durability — Cohort Overview")
    st.markdown(
        "This dashboard explores the deidentified clinical cohort behind the project "
        "(labs, medications and clinical notes, rolled up into a patient-year multimodal "
        "table) and summarizes the trained Keras survival model used to estimate "
        "**Structural Valve Deterioration (SVD) / valve failure risk** over time."
    )

    n_patients = multi["Patient"].nunique()
    n_years = len(multi)
    failure_last = multi.sort_values("Year").groupby("Patient").tail(1)["Valve_Failure"]
    fail_rate = failure_last.mean(skipna=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Patients (cohort)", f"{n_patients:,}")
    c2.metric("Patient-years", f"{n_years:,}")
    c3.metric("Valve failure rate", f"{fail_rate:.0%}")
    c4.metric("Patients w/ detailed labs & meds", f"{len(RICH_PATIENTS)}")
    c5.metric("Clinical notes", f"{len(notes):,}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Lab results", f"{len(labs):,}")
    c2.metric("Medication orders", f"{len(meds):,}")
    c3.metric("Year span", f"{int(multi['Year'].min())}–{int(multi['Year'].max())}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Valve failure outcome (latest year per patient)")
        vc = failure_last.map({0: "No failure", 1: "Failure"}).value_counts()
        fig = px.pie(
            names=vc.index,
            values=vc.values,
            hole=0.55,
            color=vc.index,
            color_discrete_map={"No failure": STATUS["good"], "Failure": STATUS["critical"]},
        )
        fig.update_traces(textinfo="label+percent")
        st.plotly_chart(style(fig, legend=False), width='stretch')

    with col2:
        st.subheader("Clinical concept mentions in notes")
        mention_sums = multi[MENTION_COLS].sum().sort_values(ascending=False)
        mention_sums.index = [c.replace("note_mentions_", "").replace("_", " ") for c in mention_sums.index]
        fig = px.bar(
            x=mention_sums.values,
            y=mention_sums.index,
            orientation="h",
            labels={"x": "Patient-years mentioning concept", "y": ""},
        )
        fig.update_traces(marker_color=CAT[0])
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(style(fig, legend=False), width='stretch')

    st.divider()
    st.subheader("Data sources")
    st.markdown(
        """
| Source | Rows | Grain | Notes |
|---|---|---|---|
| `labs_cleaned_flagged.csv` | {labs_n:,} | one lab result | 17 patients with detailed lab panels; abnormal-result flag included |
| `medications_cleaned_flagged.csv` | {meds_n:,} | one medication order | same 17 patients; therapeutic/pharmaceutical class, dose, route |
| `notes_cleaned.csv` | {notes_n:,} | one clinical note | full note text, type, service, authoring provider |
| `multimodal_patient_year_cleaned.csv` | {multi_n:,} | one patient × year | rolls up labs/meds/notes per patient-year, plus the `Valve_Failure` label used for modeling |
        """.format(
            labs_n=len(labs), meds_n=len(meds), notes_n=len(notes), multi_n=len(multi)
        )
    )
    st.caption(
        "Raw (pre-cleaning) deidentified exports also live under `data/Deidentified Dataset/` "
        "(`labs_deidentified.xlsx`, `medications_deidentified.xlsx`, `notes_deidentified.xlsx`)."
    )


# --------------------------------------------------------------------------- #
# Labs
# --------------------------------------------------------------------------- #

def page_labs():
    st.title("Laboratory Results")
    st.caption(f"{len(labs):,} results across {labs['Patient'].nunique()} patients with detailed lab panels.")

    abnormal = labs["Is Abnormal"].dropna()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lab results", f"{len(labs):,}")
    c2.metric("Unique lab components", f"{labs['Lab Component Name'].nunique():,}")
    c3.metric("Flagged abnormal", f"{abnormal.mean():.0%}" if len(abnormal) else "n/a")
    c4.metric("Extreme-value flags", f"{int(labs['robust_extreme_flag'].sum()):,}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Most frequent lab components")
        top = labs["Lab Component Name"].value_counts().head(15).sort_values()
        fig = px.bar(x=top.values, y=top.index, orientation="h", labels={"x": "Results", "y": ""})
        fig.update_traces(marker_color=CAT[0])
        st.plotly_chart(style(fig, legend=False), width='stretch')

    with col2:
        st.subheader("Results per year")
        by_year = labs["Result_Year"].value_counts().sort_index()
        fig = px.bar(x=by_year.index, y=by_year.values, labels={"x": "Year", "y": "Results"})
        fig.update_traces(marker_color=CAT[2])
        st.plotly_chart(style(fig, legend=False), width='stretch')

    st.divider()
    st.subheader("Explore a lab component")
    top_components = labs["Lab Component Name"].value_counts().head(40).index.tolist()
    default_ix = top_components.index("Creatinine") if "Creatinine" in top_components else 0
    component = st.selectbox("Lab component", top_components, index=default_ix)

    sub = labs[(labs["Lab Component Name"] == component) & labs["Numeric Value"].notna()].copy()
    if sub.empty:
        st.info("No numeric values recorded for this component.")
    else:
        unit = sub["Unit"].dropna().iloc[0] if sub["Unit"].notna().any() else ""
        sub["Abnormal"] = sub["Is Abnormal"].map({1.0: "Abnormal", 0.0: "Normal"}).fillna("Unflagged")
        fig = px.histogram(
            sub,
            x="Numeric Value",
            color="Abnormal",
            nbins=40,
            color_discrete_map={"Normal": CAT[0], "Abnormal": STATUS["critical"], "Unflagged": AXIS},
            labels={"Numeric Value": f"{component} ({unit})" if unit else component},
        )
        st.plotly_chart(style(fig, legend=True), width='stretch')

    with st.expander("Browse raw lab results"):
        patient_choice = st.selectbox("Patient", ["All"] + sorted(labs["Patient"].unique().tolist()))
        view = labs if patient_choice == "All" else labs[labs["Patient"] == patient_choice]
        view = view.sort_values(["Patient", "Result Date"])
        st.dataframe(
            view[
                [
                    "Patient",
                    "Lab Component Name",
                    "Result Date",
                    "String Value",
                    "Numeric Value",
                    "Unit",
                    "Is Abnormal",
                ]
            ],
            width='stretch',
            height=350,
        )


# --------------------------------------------------------------------------- #
# Medications
# --------------------------------------------------------------------------- #

def page_medications():
    st.title("Medications")
    st.caption(f"{len(meds):,} medication orders across {meds['Patient'].nunique()} patients with detailed medication data.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Medication orders", f"{len(meds):,}")
    c2.metric("Unique generic medications", f"{meds['Simple Generic Name'].nunique():,}")
    c3.metric("Therapeutic classes", f"{meds['Medication Therapeutic Class'].nunique():,}")
    c4.metric("Exact duplicate rows flagged", f"{int(meds['exact_duplicate_group'].sum()):,}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top therapeutic classes")
        top = meds["Medication Therapeutic Class"].value_counts().head(12).sort_values()
        fig = px.bar(x=top.values, y=top.index, orientation="h", labels={"x": "Orders", "y": ""})
        fig.update_traces(marker_color=CAT[0])
        st.plotly_chart(style(fig, legend=False), width='stretch')

    with col2:
        st.subheader("Top generic medications")
        top = meds["Simple Generic Name"].value_counts().head(12).sort_values()
        fig = px.bar(x=top.values, y=top.index, orientation="h", labels={"x": "Orders", "y": ""})
        fig.update_traces(marker_color=CAT[1])
        st.plotly_chart(style(fig, legend=False), width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Route of administration")
        top = meds["Medication Expected Route"].value_counts().head(10).sort_values()
        fig = px.bar(x=top.values, y=top.index, orientation="h", labels={"x": "Orders", "y": ""})
        fig.update_traces(marker_color=CAT[2])
        st.plotly_chart(style(fig, legend=False), width='stretch')

    with col2:
        st.subheader("Orders started per year")
        by_year = meds["Start Date_Year"].value_counts().sort_index()
        fig = px.bar(x=by_year.index, y=by_year.values, labels={"x": "Year", "y": "Orders"})
        fig.update_traces(marker_color=CAT[3])
        st.plotly_chart(style(fig, legend=False), width='stretch')

    st.divider()
    st.subheader("Cardiovascular drug-class coverage (patient-years, full cohort)")
    st.caption("From the engineered `med__*__present` flags in the multimodal table.")
    flag_rate = multi[MED_FLAG_COLS].mean().sort_values(ascending=False)
    flag_rate.index = [c.replace("med__", "").replace("__present", "").replace("_", " ") for c in flag_rate.index]
    fig = px.bar(x=flag_rate.values, y=flag_rate.index, orientation="h", labels={"x": "Share of patient-years", "y": ""})
    fig.update_traces(marker_color=CAT[0])
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(style(fig, legend=False), width='stretch')

    with st.expander("Browse raw medication orders"):
        patient_choice = st.selectbox("Patient", ["All"] + sorted(meds["Patient"].unique().tolist()), key="meds_patient")
        view = meds if patient_choice == "All" else meds[meds["Patient"] == patient_choice]
        view = view.sort_values(["Patient", "Start Date"])
        st.dataframe(
            view[
                [
                    "Patient",
                    "Proper Name",
                    "Simple Generic Name",
                    "Medication Therapeutic Class",
                    "Start Date",
                    "Frequency",
                    "Route",
                    "Dose",
                    "Dose Unit",
                ]
            ],
            width='stretch',
            height=350,
        )


# --------------------------------------------------------------------------- #
# Clinical notes
# --------------------------------------------------------------------------- #

def page_notes():
    st.title("Clinical Notes")
    st.caption(f"{len(notes):,} notes across {notes['Profile Key'].nunique()} patients.")

    word_counts = notes["Notes"].dropna().astype(str).str.split().str.len()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Notes", f"{len(notes):,}")
    c2.metric("Note types", f"{notes['Type'].nunique():,}")
    c3.metric("Median note length", f"{int(word_counts.median()):,} words")
    c4.metric("Signed notes", f"{(notes['Signed Status'] == 'Signed').mean():.0%}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Note type")
        vc = notes["Type"].value_counts().sort_values()
        fig = px.bar(x=vc.values, y=vc.index, orientation="h", labels={"x": "Notes", "y": ""})
        fig.update_traces(marker_color=CAT[0])
        st.plotly_chart(style(fig, legend=False), width='stretch')

    with col2:
        st.subheader("Authoring provider specialty")
        vc = notes["Authoring Provider Specialty"].value_counts().head(10).sort_values()
        fig = px.bar(x=vc.values, y=vc.index, orientation="h", labels={"x": "Notes", "y": ""})
        fig.update_traces(marker_color=CAT[2])
        st.plotly_chart(style(fig, legend=False), width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Notes authored per year")
        by_year = notes["Creation Date_Year"].value_counts().sort_index()
        fig = px.bar(x=by_year.index, y=by_year.values, labels={"x": "Year", "y": "Notes"})
        fig.update_traces(marker_color=CAT[3])
        st.plotly_chart(style(fig, legend=False), width='stretch')

    with col2:
        st.subheader("Note length distribution")
        fig = px.histogram(word_counts, nbins=30, labels={"value": "Words per note"})
        fig.update_traces(marker_color=CAT[0])
        st.plotly_chart(style(fig, legend=False), width='stretch')

    st.divider()
    st.subheader("Clinical concept mentions (cohort-wide, from patient-year table)")
    mention_sums = multi[MENTION_COLS].sum().sort_values(ascending=False)
    mention_sums.index = [c.replace("note_mentions_", "").replace("_", " ") for c in mention_sums.index]
    fig = px.bar(x=mention_sums.index, y=mention_sums.values, labels={"x": "", "y": "Patient-years"})
    fig.update_traces(marker_color=CAT[4])
    st.plotly_chart(style(fig, legend=False), width='stretch')

    with st.expander("Read a note"):
        idx = st.selectbox(
            "Choose a note",
            notes.index,
            format_func=lambda i: f"{notes.loc[i, 'Profile Key']} — {notes.loc[i, 'Type']} ({int(notes.loc[i, 'Creation Date_Year'])})",
        )
        st.text_area("Note text", notes.loc[idx, "Notes"], height=300)


# --------------------------------------------------------------------------- #
# Multimodal cohort
# --------------------------------------------------------------------------- #

def page_multimodal():
    st.title("Multimodal Patient-Year Cohort")
    st.caption(
        "The engineered table combining labs, medications and notes at the patient-year level — "
        "this is the row grain used to build model-ready sequences."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", f"{multi['Patient'].nunique():,}")
    c2.metric("Patient-years", f"{len(multi):,}")
    c3.metric("Avg. years / patient", f"{len(multi) / multi['Patient'].nunique():.1f}")
    c4.metric("Rows with rich lab+med data", f"{int(multi['rich_lab_med'].sum(skipna=True)):,}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Valve failure rate by year")
        by_year = multi.groupby("Year")["Valve_Failure"].mean().dropna()
        fig = px.line(x=by_year.index, y=by_year.values, markers=True, labels={"x": "Year", "y": "Failure rate"})
        fig.update_traces(line_color=CAT[7], marker_color=CAT[7])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style(fig, legend=False), width='stretch')

    with col2:
        st.subheader("Notes per patient-year")
        fig = px.histogram(multi, x="note_count", nbins=20, labels={"note_count": "Notes"})
        fig.update_traces(marker_color=CAT[0])
        st.plotly_chart(style(fig, legend=False), width='stretch')

    st.divider()
    st.subheader("Lab feature missingness")
    st.caption("Only the 17 patients with detailed lab panels populate these columns — shown here for transparency.")
    miss = (multi[LAB_COLS].isna().mean() * 100).sort_values(ascending=False)
    miss.index = [c.replace("lab__", "") for c in miss.index]
    fig = px.bar(x=miss.values, y=miss.index, orientation="h", labels={"x": "% missing", "y": ""})
    fig.update_traces(marker=dict(color=miss.values, colorscale=SEQ_BLUE))
    fig.update_layout(height=650)
    st.plotly_chart(style(fig, legend=False), width='stretch')

    st.subheader("Correlation between key labs")
    key_labs = [
        "lab__LVEF", "lab__Creatinine", "lab__eGFR", "lab__Hemoglobin",
        "lab__NT-proBNP", "lab__Troponin T", "lab__Sodium", "lab__Potassium",
    ]
    key_labs = [c for c in key_labs if c in multi.columns]
    corr = multi[key_labs].corr()
    corr.index = corr.columns = [c.replace("lab__", "") for c in corr.columns]
    fig = px.imshow(corr, color_continuous_scale=DIVERGING, zmin=-1, zmax=1, text_auto=".2f")
    fig.update_layout(height=500)
    st.plotly_chart(style(fig, legend=False), width='stretch')

    with st.expander("Browse the patient-year table"):
        st.dataframe(multi, width='stretch', height=350)


# --------------------------------------------------------------------------- #
# Patient explorer
# --------------------------------------------------------------------------- #

def page_patient_explorer():
    st.title("Patient Explorer")

    patients = sorted(multi["Patient"].unique().tolist())
    selected = st.selectbox(
        "Patient",
        patients,
        format_func=lambda p: f"{p}  {'(detailed labs/meds)' if p in RICH_PATIENTS else ''}",
    )

    pdata = multi[multi["Patient"] == selected].sort_values("Year")
    latest = pdata.iloc[-1]

    st.subheader(f"Timeline — {selected}")
    c1, c2, c3, c4 = st.columns(4)
    lvef = latest.get("lab__LVEF")
    creat = latest.get("lab__Creatinine")
    c1.metric("Latest LVEF", f"{lvef:.1f}%" if pd.notna(lvef) else "n/a")
    c2.metric("Latest creatinine", f"{creat:.2f} mg/dL" if pd.notna(creat) else "n/a")
    c3.metric("Total notes", f"{int(pdata['note_count'].sum(skipna=True)):,}")
    fail = latest.get("Valve_Failure")
    c4.metric("Valve failure (latest year)", "Yes" if fail == 1 else ("No" if fail == 0 else "n/a"))

    if fail == 1:
        st.error("High risk — evidence of Structural Valve Deterioration (SVD) or repeat intervention.")
    elif fail == 0:
        st.success("No recorded valve failure in the latest observed year.")

    col1, col2 = st.columns(2)
    with col1:
        if "lab__LVEF" in pdata.columns and pdata["lab__LVEF"].notna().any():
            fig = px.line(pdata, x="Year", y="lab__LVEF", markers=True, labels={"lab__LVEF": "LVEF (%)"})
            fig.update_traces(line_color=CAT[0], marker_color=CAT[0])
            st.plotly_chart(style(fig, legend=False), width='stretch')
        else:
            st.info("No LVEF values recorded for this patient.")
    with col2:
        if "lab__Creatinine" in pdata.columns and pdata["lab__Creatinine"].notna().any():
            fig = px.line(pdata, x="Year", y="lab__Creatinine", markers=True, labels={"lab__Creatinine": "Creatinine (mg/dL)"})
            fig.update_traces(line_color=CAT[7], marker_color=CAT[7])
            st.plotly_chart(style(fig, legend=False), width='stretch')
        else:
            st.info("No creatinine values recorded for this patient.")

    st.subheader("Year-by-year record")
    show_cols = ["Year", "Valve_Failure", "note_count", "note_types"] + [
        c for c in ["lab__LVEF", "lab__Creatinine", "lab__eGFR", "lab__NT-proBNP"] if c in pdata.columns
    ]
    st.dataframe(pdata[show_cols].set_index("Year"), width='stretch')

    med_flags = pdata[MED_FLAG_COLS].max(numeric_only=True).dropna()
    med_flags = med_flags[med_flags > 0]
    if not med_flags.empty:
        st.subheader("Medication classes ever recorded")
        st.write(", ".join(sorted(c.replace("med__", "").replace("__present", "").replace("_", " ") for c in med_flags.index)))

    if selected in RICH_PATIENTS:
        with st.expander("Raw labs for this patient"):
            st.dataframe(
                labs[labs["Patient"] == selected][
                    ["Lab Component Name", "Result Date", "String Value", "Numeric Value", "Unit", "Is Abnormal"]
                ],
                width='stretch',
                height=300,
            )
        with st.expander("Raw medications for this patient"):
            st.dataframe(
                meds[meds["Patient"] == selected][
                    ["Proper Name", "Medication Therapeutic Class", "Start Date", "Frequency", "Route", "Dose", "Dose Unit"]
                ],
                width='stretch',
                height=300,
            )


# --------------------------------------------------------------------------- #
# Risk estimator (transparent cohort-trained proxy model)
# --------------------------------------------------------------------------- #
#
# The trained Keras survival network (see the "Keras Model" page) cannot be run
# live here: it expects preprocessing (a StandardScaler + OneHotEncoder fitted
# on a 1,000-patient synthetic training split, plus fixed feature ordering)
# that was never saved to this repository -- it only exists transiently inside
# the training notebooks. Feeding it our cohort's raw columns without that
# exact pipeline would silently produce meaningless numbers.
#
# Instead, this page fits a small, fully transparent logistic-regression model
# on our actual clinical cohort (`multimodal_patient_year_cleaned.csv`) right
# here in the app, and lets you enter a patient's values to get a probability
# of Valve_Failure for that patient-year. It is an illustrative proxy, not the
# deep model -- see the caveats printed on the page.

RISK_NUMERIC_FEATURES = [
    "lab__LVEF",
    "lab__Creatinine",
    "lab__eGFR",
    "lab__NT-proBNP",
    "lab__Troponin T",
    "lab__Hemoglobin",
    "lab__INR",
    "lab__Sodium",
    "lab__Potassium",
    "note_count",
]

RISK_MED_FLAGS = [
    "med__beta_blocker__present",
    "med__ace_inhibitor__present",
    "med__arb__present",
    "med__arni__present",
    "med__loop_diuretic__present",
    "med__thiazide_diuretic__present",
    "med__mra__present",
    "med__anticoagulant__present",
    "med__antiplatelet__present",
    "med__statin__present",
    "med__ccb__present",
    "med__sglt2_inhibitor__present",
    "med__digoxin__present",
    "med__antiarrhythmic__present",
    "med__insulin__present",
]

RISK_NOTE_FLAGS = [
    "note_mentions_tavr",
    "note_mentions_savr_or_avr",
    "note_mentions_redo",
    "note_mentions_valve_in_valve",
    "note_mentions_prosthetic_failure",
    "note_mentions_endocarditis",
    "note_mentions_bioprosthetic",
    "note_mentions_mechanical",
    "note_mentions_cabg",
]

RISK_BINARY_FEATURES = RISK_MED_FLAGS + RISK_NOTE_FLAGS + ["labs_available", "medications_available", "notes_available"]

LAB_FIELD_INFO = {
    "lab__LVEF": ("LVEF", "%"),
    "lab__Creatinine": ("Creatinine", "mg/dL"),
    "lab__eGFR": ("eGFR", "mL/min/1.73m2"),
    "lab__NT-proBNP": ("NT-proBNP", "pg/mL"),
    "lab__Troponin T": ("Troponin T", "ng/mL"),
    "lab__Hemoglobin": ("Hemoglobin", "g/dL"),
    "lab__INR": ("INR", "ratio"),
    "lab__Sodium": ("Sodium", "mmol/L"),
    "lab__Potassium": ("Potassium", "mmol/L"),
}


@st.cache_resource(show_spinner=False)
def train_risk_model(df: pd.DataFrame):
    data = df[df["Valve_Failure"].notna()].copy()
    y = data["Valve_Failure"].astype(int)

    num = data[RISK_NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    medians = num.median()
    num_filled = num.fillna(medians)
    means = num_filled.mean()
    stds = num_filled.std(ddof=0).replace(0, 1.0)
    num_scaled = (num_filled - means) / stds

    binary = data[RISK_BINARY_FEATURES].fillna(0.0).clip(0, 1)

    X = pd.concat([num_scaled, binary], axis=1)
    feature_order = X.columns.tolist()

    model = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
    model.fit(X.to_numpy(), y.to_numpy())

    train_pred = model.predict_proba(X.to_numpy())[:, 1]
    return {
        "model": model,
        "feature_order": feature_order,
        "medians": medians,
        "means": means,
        "stds": stds,
        "baseline_rate": float(y.mean()),
        "n_rows": int(len(data)),
        "n_events": int(y.sum()),
        "in_sample_auc": float(roc_auc_score(y, train_pred)),
    }


def predict_risk(artifact: dict, raw: dict) -> tuple[float, pd.Series]:
    num_values = []
    for f in RISK_NUMERIC_FEATURES:
        v = raw.get(f)
        if v is None:
            v = artifact["medians"][f]
        num_values.append((v - artifact["means"][f]) / artifact["stds"][f])
    bin_values = [float(bool(raw.get(f, 0))) for f in RISK_BINARY_FEATURES]

    x = np.array(num_values + bin_values).reshape(1, -1)
    prob = float(artifact["model"].predict_proba(x)[0, 1])
    contributions = pd.Series(artifact["model"].coef_[0] * x[0], index=artifact["feature_order"])
    return prob, contributions


def _risk_band(prob: float) -> tuple[str, str]:
    if prob < 0.15:
        return "Lower estimated risk", STATUS["good"]
    if prob < 0.35:
        return "Moderate estimated risk", STATUS["warning"]
    return "Higher estimated risk", STATUS["critical"]


def page_risk_estimator():
    st.title("Patient Risk Estimator")
    st.markdown(
        "Enter a patient's values from the fields available in this cohort's data and get an "
        "estimated probability of `Valve_Failure` for that patient-year."
    )
    artifact = train_risk_model(multi)

    st.caption(
        f"Fitted on {artifact['n_rows']} patient-year rows ({artifact['n_events']} with Valve_Failure=1, "
        f"cohort base rate {artifact['baseline_rate']:.0%}). In-sample AUC {artifact['in_sample_auc']:.2f} "
        "(training-set fit, not cross-validated -- the cohort is too small to hold out a reliable test split)."
    )

    st.divider()
    st.subheader("1. Start from an existing patient (optional)")
    patient_options = ["-- custom / blank patient --"] + sorted(multi["Patient"].unique().tolist())
    source_patient = st.selectbox("Prefill from patient's latest year", patient_options)

    if source_patient != "-- custom / blank patient --":
        prefill = multi[multi["Patient"] == source_patient].sort_values("Year").iloc[-1]
    else:
        prefill = pd.Series(dtype=object)

    st.divider()
    st.subheader("2. Patient values")
    col1, col2, col3 = st.columns(3)

    entered_numeric = {}
    with col1:
        st.markdown("**Labs**")
        for f in RISK_NUMERIC_FEATURES[:-1]:
            label, unit = LAB_FIELD_INFO[f]
            default = prefill.get(f) if source_patient != "-- custom / blank patient --" else None
            default = float(default) if pd.notna(default) else float(artifact["medians"][f])
            entered_numeric[f] = st.number_input(f"{label} ({unit})", value=round(default, 2), key=f"num_{f}_{source_patient}")

    with col2:
        st.markdown("**Medications (current)**")
        entered_meds = {}
        for f in RISK_MED_FLAGS:
            label = f.replace("med__", "").replace("__present", "").replace("_", " ")
            default = bool(prefill.get(f)) if source_patient != "-- custom / blank patient --" and pd.notna(prefill.get(f)) else False
            entered_meds[f] = st.checkbox(label, value=default, key=f"med_{f}_{source_patient}")

    with col3:
        st.markdown("**This year's clinical notes mention...**")
        entered_notes = {}
        for f in RISK_NOTE_FLAGS:
            label = f.replace("note_mentions_", "").replace("_", " ")
            default = bool(prefill.get(f)) if source_patient != "-- custom / blank patient --" and pd.notna(prefill.get(f)) else False
            entered_notes[f] = st.checkbox(label, value=default, key=f"note_{f}_{source_patient}")
        default_notes = prefill.get("note_count") if source_patient != "-- custom / blank patient --" else None
        default_notes = int(default_notes) if pd.notna(default_notes) else int(artifact["medians"]["note_count"])
        entered_numeric["note_count"] = st.number_input(
            "Number of notes this year", min_value=0, value=default_notes, step=1, key=f"notecount_{source_patient}"
        )

    raw = dict(entered_numeric)
    raw.update(entered_meds)
    raw.update(entered_notes)
    raw["labs_available"] = 1.0
    raw["medications_available"] = 1.0
    raw["notes_available"] = 1.0

    st.divider()
    st.subheader("3. Estimated risk")
    prob, contributions = predict_risk(artifact, raw)
    band_label, band_color = _risk_band(prob)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric(
            "Estimated probability of valve failure (this year)",
            f"{prob:.0%}",
            delta=f"{prob - artifact['baseline_rate']:+.0%} vs cohort base rate",
            delta_color="inverse",
        )
        st.markdown(f"<span style='color:{band_color}; font-weight:600'>{band_label}</span>", unsafe_allow_html=True)

    with c2:
        top = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(10)
        top_labels = [c.replace("lab__", "").replace("med__", "").replace("note_mentions_", "").replace("__present", "").replace("_", " ") for c in top.index]
        colors = [STATUS["critical"] if v > 0 else CAT[0] for v in top.values]
        fig = go.Figure(go.Bar(x=top.values, y=top_labels, orientation="h", marker_color=colors))
        fig.update_layout(
            title="Top factors pushing risk up (red) or down (blue) for this patient",
            xaxis_title="Contribution to risk (log-odds)",
        )
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(style(fig, legend=False), width="stretch")


# --------------------------------------------------------------------------- #
# Keras model
# --------------------------------------------------------------------------- #

ARCHITECTURE_DOT = """
digraph G {
  rankdir=LR;
  bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#c3c2b7"];
  edge [color="#898781", arrowsize=0.7];

  static_input [label="static_input\\n(13 valve/static features)", fillcolor="#cde2fb", fontcolor="#0b0b0b"];
  phys_input [label="phys_input\\n(sequence x 71 physiology/lab features)", fillcolor="#cde2fb", fontcolor="#0b0b0b"];
  med_input [label="med_input\\n(sequence x 31 medication features)", fillcolor="#cde2fb", fontcolor="#0b0b0b"];

  dense_static [label="Dense 64 (relu)\\n+ Dropout 0.15", fillcolor="#eda100", fontcolor="#0b0b0b"];
  z_valve [label="z_valve\\nDense 32 (relu)", fillcolor="#eda100", fontcolor="#0b0b0b"];

  masking_phys [label="Masking (0.0)", fillcolor="#e1e0d9", fontcolor="#0b0b0b"];
  phys_gru [label="phys_gru\\nGRU(64)", fillcolor="#1baf7a", fontcolor="#0b0b0b"];
  z_phys [label="z_phys\\nDense 32 (relu)", fillcolor="#1baf7a", fontcolor="#0b0b0b"];

  masking_med [label="Masking (0.0)", fillcolor="#e1e0d9", fontcolor="#0b0b0b"];
  med_gru [label="med_gru\\nGRU(48)", fillcolor="#4a3aa7", fontcolor="#ffffff"];
  z_med [label="z_med\\nDense 32 (relu)", fillcolor="#4a3aa7", fontcolor="#ffffff"];

  concat1 [label="Concatenate", shape=diamond, fillcolor="#f0efec", fontcolor="#0b0b0b"];
  temporal_dense [label="Dense 64 (relu)", fillcolor="#2a78d6", fontcolor="#ffffff"];
  z_patient [label="z_patient\\nDense 64 (relu)", fillcolor="#2a78d6", fontcolor="#ffffff"];

  concat2 [label="Concatenate", shape=diamond, fillcolor="#f0efec", fontcolor="#0b0b0b"];
  fused_dense [label="Dense 64 (relu)", fillcolor="#e34948", fontcolor="#ffffff"];
  hazards [label="hazards\\nDense 10 (sigmoid)\\n= yearly hazard, year 1..10", fillcolor="#e34948", fontcolor="#ffffff"];

  static_input -> dense_static -> z_valve;
  phys_input -> masking_phys -> phys_gru -> z_phys;
  med_input -> masking_med -> med_gru -> z_med;
  z_phys -> concat1; z_med -> concat1;
  concat1 -> temporal_dense -> z_patient;
  z_patient -> concat2; z_valve -> concat2;
  concat2 -> fused_dense -> hazards;
}
"""

EVAL_METRICS = [
    ("Harrell C-index (5y risk)", 0.8799, "higher is better"),
    ("Time-dependent AUC @5y", 0.9708, "higher is better"),
    ("Brier score @2y", 0.0008, "lower is better"),
    ("Brier score @5y", 0.0380, "lower is better"),
    ("Brier score @8y", 0.1127, "lower is better"),
    ("Integrated Brier Score", 0.0529, "lower is better"),
]

ABLATION = pd.DataFrame(
    [
        {"model": "Valve only", "c_index": 0.4749, "brier_5y": 0.0902, "ibs": 0.0954, "auc_5y": 0.4814},
        {"model": "Physiology only", "c_index": 0.8788, "brier_5y": 0.0446, "ibs": 0.0548, "auc_5y": 0.9671},
        {"model": "Medications only", "c_index": 0.8006, "brier_5y": 0.0614, "ibs": 0.0622, "auc_5y": 0.9317},
        {"model": "Valve + Physiology", "c_index": 0.8695, "brier_5y": 0.0393, "ibs": 0.0550, "auc_5y": 0.9702},
        {"model": "Full multimodal", "c_index": 0.8756, "brier_5y": 0.0457, "ibs": 0.0553, "auc_5y": 0.9677},
    ]
)


def page_model():
    st.title("Keras Multimodal Survival Model")
    st.markdown(
        "The model is a **discrete-time survival network**: three input branches "
        "(static valve/patient features, a yearly physiology/labs sequence, and a "
        "yearly medications sequence) feed two GRUs and a small MLP, which fuse into "
        "10 sigmoid **yearly hazard** outputs (year 1 through year 10 post-implant). "
        "A predicted survival curve is `S(t) = Π(1 − hazard_k)` over the elapsed yearly bins."
    )
    st.info(
        "The model was trained and evaluated on a larger **synthetic** multimodal cohort "
        "(1,000 patients, 11 yearly timesteps, 7,169 patient-year rows) built specifically for "
        "model development — see `ml/Encoding Files/`. This differs from the smaller deidentified "
        "clinical cohort explored in the other tabs. Because the exact preprocessing (scaler/encoder, "
        "feature ordering) used to build this model's inputs was never saved to this repository, it "
        "cannot be run live from this dashboard. For a live, adjustable prediction on our own cohort, "
        "see the **Risk Estimator** page -- a transparent logistic-regression proxy, not this network."
    )

    arch = load_model_architecture(MODEL_PATH)
    if arch is None:
        st.error(f"Model file not found at `{MODEL_PATH}`.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total parameters", f"{arch['total_params']:,}")
    c2.metric("Layers", f"{len(arch['layers'])}")
    c3.metric("Keras version", arch["keras_version"] or "n/a")
    c4.metric("Hazard bins", "10 (0–10y, 1y steps)")

    st.subheader("Architecture")
    st.graphviz_chart(ARCHITECTURE_DOT, width='stretch')

    with st.expander("Layer-by-layer breakdown (parsed from the .keras weights file)"):
        df_layers = pd.DataFrame(arch["layers"])
        st.dataframe(df_layers, width='stretch', hide_index=True)

    tc = arch.get("training_config", {})
    opt = tc.get("optimizer_config", {}).get("config", {})
    st.subheader("Training configuration")
    c1, c2, c3 = st.columns(3)
    c1.metric("Loss", tc.get("loss", "n/a"))
    c2.metric("Optimizer", tc.get("optimizer_config", {}).get("class_name", "n/a"))
    c3.metric("Learning rate", f"{opt.get('learning_rate', 0):.4f}" if opt.get("learning_rate") else "n/a")
    st.caption(
        "`survival_loss` is a custom discrete-time hazard negative log-likelihood: for each patient "
        "it sums `event·log(h) + (at_risk − event)·log(1−h)` across the 10 yearly bins."
    )

    st.divider()
    st.subheader("Reported evaluation metrics")
    st.caption("From the held-out test split (150 patients) in `ml/Encoding Files/04_survival_evaluation_tensorflow.ipynb`.")
    cols = st.columns(len(EVAL_METRICS))
    for col, (label, value, direction) in zip(cols, EVAL_METRICS):
        col.metric(label, f"{value:.4f}", help=direction)
    st.caption("Best validation loss 0.6378 at epoch 17 of training (Adam, survival NLL).")

    st.divider()
    st.subheader("Modality ablation study")
    st.caption(
        "Each row retrains the same architecture using only the listed input branch(es), "
        "on the same patient split — from `ml/Encoding Files/05_ablation_study_tensorflow.ipynb`."
    )
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(ABLATION, x="model", y="c_index", labels={"c_index": "C-index", "model": ""})
        fig.update_traces(marker_color=CAT[0])
        fig.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(style(fig, legend=False), width='stretch')
    with col2:
        fig = px.bar(ABLATION, x="model", y="brier_5y", labels={"brier_5y": "Brier @5y (lower is better)", "model": ""})
        fig.update_traces(marker_color=CAT[7])
        fig.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(style(fig, legend=False), width='stretch')
    st.dataframe(ABLATION, width='stretch', hide_index=True)
    st.markdown(
        "**Reading it:** physiology (labs) carries most of the discriminative signal on its own "
        "(C-index 0.879); medications alone are also informative (0.801); the static valve-only "
        "branch is close to chance (0.475). The full multimodal model does not clearly beat "
        "physiology-only on this test split — consistent with physiology dominating the signal."
    )

    st.divider()
    st.subheader("Explainability (SHAP)")
    st.markdown(
        "`ml/Encoding Files/05_shap_analysis_tensorflow.ipynb` explains the model's predicted "
        "**5-year failure risk** (`1 − S(60 months)`) with `GradientExplainer`: global mean "
        "`|SHAP|` importance per static/physiology/medication feature, an importance roll-up by "
        "modality branch, and per-patient waterfall-style explanations of which values pushed a "
        "given patient's 5-year risk up or down."
    )


# --------------------------------------------------------------------------- #
# Valve candidate comparison (v2 model)
# --------------------------------------------------------------------------- #

FOOTER_DISCLAIMER = (
    "Research prototype — data-grounded semi-synthetic proof of concept. Predictions are not "
    "clinically validated and should not be used for treatment decisions."
)

V2_ARCHITECTURE_DOT = """
digraph G {
  rankdir=LR;
  bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#c3c2b7"];
  edge [color="#898781", arrowsize=0.7];

  labs_input [label="Longitudinal labs\\n(30 features x years)", fillcolor="#cde2fb", fontcolor="#0b0b0b"];
  meds_input [label="Longitudinal medications\\n(15 features x years)", fillcolor="#cde2fb", fontcolor="#0b0b0b"];
  history_input [label="Prior valve history\\n(Qwen-extracted, 7 features)", fillcolor="#cde2fb", fontcolor="#0b0b0b"];
  candidate_input [label="Candidate valve\\n(procedure, model, size, position)", fillcolor="#cde2fb", fontcolor="#0b0b0b"];

  lab_gru [label="Lab GRU(64)", fillcolor="#1baf7a", fontcolor="#0b0b0b"];
  med_gru [label="Med GRU(48)", fillcolor="#4a3aa7", fontcolor="#ffffff"];
  history_mlp [label="History MLP", fillcolor="#eda100", fontcolor="#0b0b0b"];

  patient_proj [label="Patient projection\\nz_patient (48)", fillcolor="#2a78d6", fontcolor="#ffffff"];
  candidate_enc [label="Candidate encoder\\nz_valve (48)", fillcolor="#eb6834", fontcolor="#0b0b0b"];

  interaction [label="Interaction features:\\nz_patient, z_valve,\\nz_patient x z_valve,\\n|z_patient - z_valve|", shape=diamond, fillcolor="#f0efec", fontcolor="#0b0b0b"];
  fusion [label="Interaction fusion MLP", fillcolor="#e34948", fontcolor="#ffffff"];
  survival_head [label="Survival head\\nDense 12 (sigmoid hazards)", fillcolor="#e34948", fontcolor="#ffffff"];

  labs_input -> lab_gru -> patient_proj;
  meds_input -> med_gru -> patient_proj;
  history_input -> history_mlp -> patient_proj;
  candidate_input -> candidate_enc;
  patient_proj -> interaction;
  candidate_enc -> interaction;
  interaction -> fusion -> survival_head;
}
"""


def _v2_missing() -> bool:
    return not (V2_MODEL_DIR / "personalized_valve_survival_model.pt").exists()


@st.cache_data(show_spinner="Scoring every patient x candidate combination...")
def compute_procedure_type_comparison(_art: dict) -> pd.DataFrame:
    rows = []
    catalog = _art["candidate_catalog"]
    for pid in _art["real_patients"]:
        prepared = vm.prepare_real_patient_trajectory(_art, pid)
        history_dict, _ = vm.prior_history_from_qwen(_art, pid)
        history_tensor = vm.transform_history_vector(_art, history_dict)
        for _, cand in catalog.iterrows():
            pred = vm.predict_candidate(_art, prepared, history_tensor, cand)
            rows.append(
                {
                    "patient": pid,
                    "candidate_id": cand["candidate_id"],
                    "procedure_type": cand["candidate_procedure_type"],
                    "rmst_years": pred["rmst_years"],
                    "survival_5y": pred["survival_5y"],
                    "survival_8y": pred["survival_8y"],
                    "survival_10y": pred["survival_10y"],
                }
            )
    return pd.DataFrame(rows)


def page_valve_comparison():
    st.title("Valve Candidate Comparison (v2)")
    st.markdown(
        "For one real pre-operative patient, compare **model-estimated event-free durability** "
        "across multiple candidate prosthetic valves (procedure, model, size). Uses the frozen "
        "interaction-aware survival model and the patient's longitudinal labs, medications, and "
        "Qwen-extracted prior valve history."
    )

    if _v2_missing():
        st.error(f"v2 model bundle not found at `{V2_MODEL_DIR}`.")
        return

    art = load_v2_artifacts(V2_MODEL_DIR, V2_DATA_DIR)

    st.subheader("1. Patient")
    patient_id = st.selectbox("Real donor patient", art["real_patients"])

    prepared = vm.prepare_real_patient_trajectory(art, patient_id)
    history_dict, readable_history = vm.prior_history_from_qwen(art, patient_id)
    history_tensor = vm.transform_history_vector(art, history_dict)

    c1, c2, c3 = st.columns(3)
    c1.metric("Longitudinal observations (years)", int(prepared["lab"].shape[0]))
    c2.metric("Hypothetical procedure year", f"{int(prepared['surgery_year'])}")
    c3.metric("Prior valve procedures", int(history_dict["prior_valve_count"]))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Pre-operative lab trajectory**")
        src = prepared["source_table"]
        avail_labs = [c for c in ["lab__LVEF", "lab__Creatinine", "lab__eGFR", "lab__NT-proBNP"] if c in src.columns and src[c].notna().any()]
        if avail_labs:
            plot_df = src[["Year"] + avail_labs].melt("Year", var_name="lab", value_name="value").dropna()
            plot_df["lab"] = plot_df["lab"].str.replace("lab__", "")
            fig = px.line(plot_df, x="Year", y="value", color="lab", markers=True, color_discrete_sequence=CAT)
            st.plotly_chart(style(fig, legend=True), width="stretch")
        else:
            st.info("No key lab values recorded for this patient.")

    with col2:
        st.markdown("**Qwen-extracted prior valve history**")
        if readable_history["procedure_types"]:
            for i, ptype in enumerate(readable_history["procedure_types"]):
                model_name = readable_history["valve_models"][i] if i < len(readable_history["valve_models"]) else "unknown model"
                size = readable_history["valve_sizes_mm"][i] if i < len(readable_history["valve_sizes_mm"]) else None
                size_txt = f", {float(size):.0f} mm" if size not in (None, "") and pd.notna(size) else ""
                st.write(f"- {ptype} — {model_name}{size_txt}")
        else:
            st.write("No prior valve procedure extracted from notes.")
        redo = "Yes" if history_dict["prior_redo_count"] > 0 else "No"
        viv = "Yes" if history_dict["prior_ViV_count"] > 0 else "No"
        st.caption(f"Redo procedure: {redo}  |  Valve-in-valve: {viv}")

    st.divider()
    st.subheader("2. Candidate valves")
    catalog = art["candidate_catalog"]
    options = catalog["candidate_id"].tolist()

    def _fmt(cid):
        row = catalog[catalog["candidate_id"] == cid].iloc[0]
        size = f"{row['candidate_valve_size_mm']:.0f}mm" if pd.notna(row["candidate_valve_size_mm"]) else "size n/a"
        return f"{cid} — {row['candidate_procedure_type']} / {row['candidate_valve_model']} / {size}"

    selected_ids = st.multiselect("Evaluate candidates", options, default=options, format_func=_fmt)

    if not selected_ids:
        st.info("Select at least one candidate valve to evaluate.")
        return

    selected = catalog[catalog["candidate_id"].isin(selected_ids)]

    st.divider()
    st.subheader("3. Model-estimated durability")

    predictions = []
    for _, cand in selected.iterrows():
        pred = vm.predict_candidate(art, prepared, history_tensor, cand)
        predictions.append(
            {
                "candidate_id": cand["candidate_id"],
                "procedure_type": cand["candidate_procedure_type"],
                "valve_model": cand["candidate_valve_model"],
                "valve_size_mm": cand["candidate_valve_size_mm"],
                "estimated_event_free_years": round(pred["rmst_years"], 2),
                "median_survival_years": round(pred["median_survival_years"], 2) if pd.notna(pred["median_survival_years"]) else None,
                "survival_5y_pct": round(100 * pred["survival_5y"], 1),
                "survival_8y_pct": round(100 * pred["survival_8y"], 1),
                "survival_10y_pct": round(100 * pred["survival_10y"], 1),
                "survival_curve": pred["survival_curve"],
            }
        )

    comparison_df = pd.DataFrame(predictions).sort_values("estimated_event_free_years", ascending=False).reset_index(drop=True)
    top = comparison_df.iloc[0]

    st.metric(
        f"Highest model-estimated durability among evaluated candidates — {top['candidate_id']} "
        f"({top['procedure_type']}, {top['valve_model']})",
        f"{top['estimated_event_free_years']:.2f} years",
        help=f"Restricted mean event-free survival time within the {art['horizon_years']:.0f}-year model horizon.",
    )

    st.dataframe(
        comparison_df.drop(columns=["survival_curve"]).rename(
            columns={
                "candidate_id": "Candidate",
                "procedure_type": "Procedure",
                "valve_model": "Model",
                "valve_size_mm": "Size (mm)",
                "estimated_event_free_years": "Est. event-free years (12y RMST)",
                "median_survival_years": "Median survival (y)",
                "survival_5y_pct": "5y event-free %",
                "survival_8y_pct": "8y event-free %",
                "survival_10y_pct": "10y event-free %",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("**Survival curves**")
    horizon = int(art["horizon_years"])
    years_axis = list(range(0, horizon + 1))
    fig = go.Figure()
    n = len(comparison_df)
    for i, row in comparison_df.iterrows():
        y = [1.0] + list(row["survival_curve"])
        if n <= 8:
            color = CAT[i % len(CAT)]
        else:
            frac = i / max(n - 1, 1)
            color = SEQ_BLUE[int(frac * (len(SEQ_BLUE) - 1))]
        fig.add_trace(go.Scatter(x=years_axis, y=y, mode="lines", name=row["candidate_id"], line=dict(color=color, width=2)))
    fig.update_layout(xaxis_title="Years since procedure", yaxis_title="Event-free survival probability", yaxis_tickformat=".0%")
    st.plotly_chart(style(fig, legend=True), width="stretch")

    st.caption(
        "Use: **model-estimated event-free durability**. This is the highest model estimate among "
        "evaluated candidates — not a \"best valve\" or a recommended treatment."
    )

    st.divider()
    st.subheader("4. Prediction drivers — why this estimate?")
    st.caption(
        f"Grouped-occlusion sensitivity relative to the top candidate ({top['candidate_id']}). "
        "These are model-sensitivity contributions, not causal effects."
    )
    reference_row = selected[selected["candidate_id"] == top["candidate_id"]].iloc[0]
    with st.spinner("Computing patient-history drivers..."):
        driver_df, _ = vm.compute_history_drivers(art, prepared, history_tensor, reference_row)
    top_drivers = driver_df.head(12).sort_values("effect_years")
    labels = top_drivers["modality"] + " | " + top_drivers["feature"]
    colors = [STATUS["critical"] if v < 0 else CAT[0] for v in top_drivers["effect_years"]]
    fig = go.Figure(go.Bar(x=top_drivers["effect_years"], y=labels, orientation="h", marker_color=colors))
    fig.update_layout(
        title="Patient factors influencing the model estimate",
        xaxis_title="Contribution to model-estimated RMST (years)",
    )
    st.plotly_chart(style(fig, legend=False), width="stretch")

    st.divider()
    st.subheader("5. Why does the estimate differ between two candidates?")
    col_a, col_b = st.columns(2)
    default_b = top["candidate_id"]
    default_a = comparison_df.iloc[1]["candidate_id"] if len(comparison_df) > 1 else default_b
    with col_a:
        cand_a_id = st.selectbox(
            "Candidate A", selected_ids, index=selected_ids.index(default_a), format_func=_fmt, key="cand_a"
        )
    with col_b:
        cand_b_id = st.selectbox(
            "Candidate B", selected_ids, index=selected_ids.index(default_b), format_func=_fmt, key="cand_b"
        )

    if cand_a_id == cand_b_id:
        st.info("Choose two different candidates to compare.")
    else:
        cand_A = catalog[catalog["candidate_id"] == cand_a_id].iloc[0]
        cand_B = catalog[catalog["candidate_id"] == cand_b_id].iloc[0]
        with st.spinner("Computing Shapley attribution..."):
            shap_df, rmst_A, rmst_B = vm.compute_candidate_shapley(art, prepared, history_tensor, cand_A, cand_B)
        st.metric(f"Predicted durability difference: {cand_b_id} − {cand_a_id}", f"{rmst_B - rmst_A:+.2f} years")
        plot_shap = shap_df.sort_values("shapley_delta_years")
        colors = [CAT[0] if v >= 0 else STATUS["critical"] for v in plot_shap["shapley_delta_years"]]
        fig = go.Figure(
            go.Bar(x=plot_shap["shapley_delta_years"], y=plot_shap["candidate_feature"], orientation="h", marker_color=colors)
        )
        fig.update_layout(
            title="Why the model predicts different durability for the two candidates",
            xaxis_title=f"Contribution to durability difference {cand_b_id} − {cand_a_id} (years)",
        )
        st.plotly_chart(style(fig, legend=False), width="stretch")
        shap_display = shap_df.copy()
        shap_display["A_value"] = shap_display["A_value"].astype(str)
        shap_display["B_value"] = shap_display["B_value"].astype(str)
        st.dataframe(
            shap_display.rename(
                columns={
                    "candidate_feature": "Feature",
                    "shapley_delta_years": "Shapley contribution (years)",
                    "A_value": f"{cand_a_id} value",
                    "B_value": f"{cand_b_id} value",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.divider()
    st.subheader("6. SAVR vs TAVR — cohort-wide model estimates")
    st.caption(
        "Every one of the 17 real donor patients evaluated against every candidate valve in the "
        "observed catalog (12 SAVR configurations, 4 TAVR configurations), grouped by procedure "
        "type. This compares the model's estimates across procedure type — it is **not** a matched "
        "or randomized clinical comparison: patients were not assigned to a procedure type, and "
        "candidates within each group also differ in model and size."
    )

    proc_df = compute_procedure_type_comparison(art)
    summary = (
        proc_df.groupby("procedure_type")
        .agg(
            n_estimates=("rmst_years", "size"),
            mean_rmst=("rmst_years", "mean"),
            median_rmst=("rmst_years", "median"),
            mean_survival_5y=("survival_5y", "mean"),
            mean_survival_8y=("survival_8y", "mean"),
            mean_survival_10y=("survival_10y", "mean"),
        )
        .reset_index()
    )
    summary_display = summary.rename(
        columns={
            "procedure_type": "Procedure",
            "n_estimates": "N estimates (patients x candidates)",
            "mean_rmst": "Mean est. event-free years",
            "median_rmst": "Median est. event-free years",
            "mean_survival_5y": "Mean 5y event-free",
            "mean_survival_8y": "Mean 8y event-free",
            "mean_survival_10y": "Mean 10y event-free",
        }
    )
    for c in ["Mean est. event-free years", "Median est. event-free years"]:
        summary_display[c] = summary_display[c].round(2)
    for c in ["Mean 5y event-free", "Mean 8y event-free", "Mean 10y event-free"]:
        summary_display[c] = (100 * summary_display[c]).round(1).astype(str) + "%"
    st.dataframe(summary_display, width="stretch", hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            proc_df, x="procedure_type", y="rmst_years", color="procedure_type",
            color_discrete_map={"SAVR": CAT[0], "TAVR": CAT[1]}, points="all",
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Model-estimated event-free years (12y RMST)",
            title="Distribution of estimates across all patients x candidates",
        )
        st.plotly_chart(style(fig, legend=False), width="stretch")

    with col2:
        melted = summary.melt(
            id_vars="procedure_type",
            value_vars=["mean_survival_5y", "mean_survival_8y", "mean_survival_10y"],
            var_name="horizon", value_name="survival",
        )
        melted["horizon"] = melted["horizon"].map({
            "mean_survival_5y": "5y", "mean_survival_8y": "8y", "mean_survival_10y": "10y",
        })
        fig = px.bar(
            melted, x="horizon", y="survival", color="procedure_type", barmode="group",
            color_discrete_map={"SAVR": CAT[0], "TAVR": CAT[1]},
        )
        fig.update_layout(xaxis_title="", yaxis_title="Mean event-free probability", yaxis_tickformat=".0%",
                           title="Mean event-free survival by horizon")
        st.plotly_chart(style(fig, legend=True), width="stretch")

    st.divider()
    st.markdown(f"**{FOOTER_DISCLAIMER}**")


# --------------------------------------------------------------------------- #
# Valve durability model (v2) — architecture, metrics, validation caveat
# --------------------------------------------------------------------------- #

def page_valve_model_v2():
    st.title("Valve Durability Model (v2)")
    st.markdown(
        "A pre-operative, patient- **and** valve-specific discrete-time survival model. A "
        "patient's longitudinal labs and medications (GRU-encoded), Qwen-extracted prior valve "
        "history, and a proposed candidate valve (procedure, model, size, position) are fused "
        "through explicit interaction features — `z_patient`, `z_valve`, their product, and their "
        "absolute difference — into 12 yearly hazards, converted to a survival curve and a "
        "12-year restricted mean survival time (RMST)."
    )

    if _v2_missing():
        st.error(f"v2 model bundle not found at `{V2_MODEL_DIR}`.")
        return

    art = load_v2_artifacts(V2_MODEL_DIR, V2_DATA_DIR)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hazard bins", f"{art['n_bins']} ({art['bin_width_years']:.0f}y each)")
    c2.metric("Horizon", f"{art['horizon_years']:.0f} years")
    c3.metric("Lab / med / history features", f"{len(art['lab_cols'])} / {len(art['med_cols'])} / {len(art['history_cols'])}")
    c4.metric("Real donor patients", f"{len(art['real_patients'])}")

    st.subheader("Architecture")
    st.graphviz_chart(V2_ARCHITECTURE_DOT, width="stretch")

    st.divider()
    st.subheader("Reported evaluation metrics")
    st.caption("Held-out synthetic patient instances from a shared 17-patient real-donor library — see validation caveat below.")
    m = art["metrics"]
    metric_rows = [
        ("Harrell C-index", f"{m['test_harrell_c_index']:.3f}"),
        ("Within-patient ranking (Spearman)", f"{m['mean_within_patient_spearman']:.3f}"),
        ("Top-1 candidate recovery", f"{m['top1_candidate_accuracy']:.1%}"),
        ("Pairwise candidate-ranking accuracy", f"{m['pairwise_candidate_ranking_accuracy']:.1%}"),
        ("Global hidden-truth Spearman", f"{m['global_hidden_truth_spearman']:.3f}"),
        ("Global hidden-truth Pearson", f"{m['global_hidden_truth_pearson']:.3f}"),
    ]
    cols = st.columns(3)
    for i, (label, value) in enumerate(metric_rows):
        cols[i % 3].metric(label, value)

    st.divider()
    st.subheader("Baseline vs. final (interaction-aware) model")
    ablation_path = V2_MODEL_DIR / "baseline_vs_final_ablation.csv"
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        ablation_display = ablation.copy()
        ablation_display["metric"] = ablation_display["metric"].str.replace("_", " ")
        st.dataframe(ablation_display, width="stretch", hide_index=True)

        plot_df = ablation.melt(id_vars="metric", value_vars=["baseline_03", "final_03b"], var_name="model", value_name="value")
        plot_df["model"] = plot_df["model"].map({"baseline_03": "Baseline", "final_03b": "Final (interaction-aware)"})
        plot_df["metric"] = plot_df["metric"].str.replace("_", " ")
        fig = px.bar(plot_df, x="metric", y="value", color="model", barmode="group", color_discrete_sequence=[AXIS, CAT[0]])
        fig.update_layout(xaxis_tickangle=-20, yaxis_title="")
        st.plotly_chart(style(fig, legend=True), width="stretch")
        st.markdown(
            "The final interaction-aware model trades a little population-level discrimination "
            "(C-index 0.878 → 0.863) for a large gain on the product-relevant question — "
            "**ranking candidate valves correctly for the same patient** "
            "(pairwise ranking accuracy 68.8% → 88.3%, top-1 recovery 51.3% → 76.0%)."
        )

    st.divider()
    st.subheader("Validation caveat")
    st.markdown(
        "The train/validation/test split is grouped by **synthetic patient identity**, so the "
        "candidate-valve scenarios for one synthetic patient never cross splits. However, the "
        "1,000 synthetic patients are bootstrapped from only **17 real donor trajectories**, and "
        "those donor identities are reused across train, validation, and test. The metrics above "
        "measure recovery of the semi-synthetic ranking task within that shared donor library — "
        "they are **not** donor-independent validation on unseen real patients."
    )

    st.divider()
    st.subheader("Real vs. semi-synthetic")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Real / data-grounded**")
        st.markdown(
            "- Longitudinal lab trajectories\n"
            "- Medication trajectories\n"
            "- Missingness patterns\n"
            "- Qwen-extracted prior valve histories\n"
            "- Observed valve procedure types, models, sizes"
        )
    with col2:
        st.markdown("**Simulated**")
        st.markdown(
            "- Long-term event time\n"
            "- Censoring\n"
            "- Counterfactual outcomes under valves not actually implanted\n"
            "- Patient × valve durability interactions"
        )
    st.caption(
        "Because the available cohort is too small to directly learn reliable long-term "
        "patient-specific valve durability, a data-grounded semi-synthetic development cohort is "
        "used: real physiology, medications, missingness, prior valve history and candidate valve "
        "characteristics, with simulated long-term durability and counterfactual responses."
    )

    st.divider()
    st.markdown(f"**{FOOTER_DISCLAIMER}**")


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #

ROUTES = {
    "Overview": page_overview,
    "Labs": page_labs,
    "Medications": page_medications,
    "Clinical Notes": page_notes,
    "Multimodal Cohort": page_multimodal,
    "Patient Explorer": page_patient_explorer,
    "Risk Estimator": page_risk_estimator,
    "Keras Model": page_model,
    "Valve Candidate Comparison (v2)": page_valve_comparison,
    "Valve Durability Model (v2)": page_valve_model_v2,
}

ROUTES[page]()
