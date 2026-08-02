"""HeartSaver — Streamlit interface for the model built in HeartSaver.ipynb.

The notebook writes ``heartsaver_model.pkl`` as a four-key dictionary containing
the fitted pipeline, selected feature list, out-of-fold decision threshold, and the
test-set results that threshold produced. This app reads that complete contract so
the model inputs, the 0.30 operating point, and the displayed metrics cannot
silently drift apart from whatever the notebook last produced.
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="HeartSaver — cardiac referral support",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(__file__).parent / "heartsaver_model.pkl"

SUPPORTED_FEATURES = {
    "ST_Slope",
    "ChestPainType",
    "Oldpeak",
    "ExerciseAngina",
    "Sex",
    "MaxHR",
    "FastingBS",
    "Age",
    "HasSTDepression",
}

REQUIRED_TEST_RESULT_KEYS = {
    "n_patients",
    "recall",
    "precision",
    "accuracy",
    "false_negatives",
    "false_positives",
}

# These ranges describe the records used by the notebook. Values outside them are
# accepted when clinically possible, but the app warns that they are extrapolations.
TRAINING_RANGES = {
    "Age": (28, 77),
    "MaxHR": (60, 202),
    "Oldpeak": (-2.6, 6.2),
}

LABELS = {
    "Sex": {"F": "Female", "M": "Male"},
    "ChestPainType": {
        "ASY": "Asymptomatic",
        "ATA": "Atypical angina",
        "NAP": "Non-anginal pain",
        "TA": "Typical angina",
    },
    "ExerciseAngina": {"N": "No", "Y": "Yes"},
    "ST_Slope": {
        "Up": "Upsloping",
        "Flat": "Flat",
        "Down": "Downsloping",
    },
    "FastingBS": {
        0: "120 mg/dL or below",
        1: "Above 120 mg/dL",
    },
}

# The first two profiles reproduce the notebook's worked examples. The third is a
# dataset profile with a model score almost exactly at the 0.30 operating point.
PRESETS = {
    "Higher-score example": {
        "Age": 62,
        "Sex": "M",
        "ChestPainType": "ASY",
        "Oldpeak": 2.4,
        "ExerciseAngina": "Y",
        "MaxHR": 110,
        "FastingBS": 1,
        "ST_Slope": "Flat",
    },
    "Lower-score example": {
        "Age": 41,
        "Sex": "F",
        "ChestPainType": "ATA",
        "Oldpeak": 0.0,
        "ExerciseAngina": "N",
        "MaxHR": 172,
        "FastingBS": 0,
        "ST_Slope": "Up",
    },
    "Near-threshold example": {
        "Age": 39,
        "Sex": "M",
        "ChestPainType": "ATA",
        "Oldpeak": 2.0,
        "ExerciseAngina": "N",
        "MaxHR": 146,
        "FastingBS": 0,
        "ST_Slope": "Up",
    },
}


def validate_bundle(candidate: Any) -> str | None:
    """Return a user-facing problem if a loaded model contract is invalid."""
    if not isinstance(candidate, dict):
        return "The model file contains a bare estimator rather than the required bundle."

    missing = {"model", "features", "threshold", "test_results"} - set(candidate)
    if missing:
        return f"The model file is missing these entries: {sorted(missing)}."

    features = candidate["features"]
    if not isinstance(features, (list, tuple)) or not features:
        return "The stored feature list is empty or invalid."

    duplicate_features = sorted({name for name in features if features.count(name) > 1})
    if duplicate_features:
        return f"The stored feature list contains duplicates: {duplicate_features}."

    unknown = sorted(set(features) - SUPPORTED_FEATURES)
    absent = sorted(SUPPORTED_FEATURES - set(features))
    if unknown or absent:
        details = []
        if unknown:
            details.append(f"unsupported features {unknown}")
        if absent:
            details.append(f"missing expected features {absent}")
        return "The app and notebook feature contracts differ: " + "; ".join(details) + "."

    try:
        threshold = float(candidate["threshold"])
    except (TypeError, ValueError):
        return "The stored decision threshold is not numeric."
    if not 0 < threshold < 1:
        return "The stored decision threshold must be between 0 and 1."

    test_results = candidate["test_results"]
    if not isinstance(test_results, dict):
        return "The stored test_results entry is not a dictionary."
    missing_metrics = REQUIRED_TEST_RESULT_KEYS - set(test_results)
    if missing_metrics:
        return f"The stored test_results is missing these entries: {sorted(missing_metrics)}."
    try:
        for key in ("recall", "precision", "accuracy"):
            float(test_results[key])
        int(test_results["n_patients"])
        int(test_results["false_negatives"])
        int(test_results["false_positives"])
    except (TypeError, ValueError):
        return "The stored test_results contains a non-numeric value."

    model = candidate["model"]
    if not hasattr(model, "predict_proba"):
        return "The stored model does not provide predict_proba()."

    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        return "The stored model does not contain the positive class labelled 1."

    return None


def patient_frame(values: dict[str, Any], features: list[str]) -> pd.DataFrame:
    """Build one model-ready row and reproduce notebook feature engineering."""
    row = dict(values)
    row["HasSTDepression"] = int(float(row["Oldpeak"]) > 0)
    return pd.DataFrame([row]).loc[:, features]


def disease_probability(bundle: dict[str, Any], values: dict[str, Any]) -> float:
    """Return the fitted model's class-1 probability for one completed form."""
    model = bundle["model"]
    features = list(bundle["features"])
    positive_column = list(model.classes_).index(1)
    probability = model.predict_proba(patient_frame(values, features))[0, positive_column]
    return float(probability)


def range_warnings(values: dict[str, Any]) -> list[str]:
    """Identify valid inputs outside the ranges represented during training."""
    warnings = []
    descriptions = {
        "Age": "age",
        "MaxHR": "maximum heart rate",
        "Oldpeak": "ST depression (Oldpeak)",
    }
    for name, (minimum, maximum) in TRAINING_RANGES.items():
        value = float(values[name])
        if not minimum <= value <= maximum:
            warnings.append(
                f"The entered {descriptions[name]} ({value:g}) is outside the training "
                f"range of {minimum:g}–{maximum:g}. Treat this estimate cautiously."
            )
    return warnings


@st.cache_resource(show_spinner="Loading the HeartSaver model…")
def load_bundle(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load and validate the notebook artefact without crashing the interface."""
    if not path.exists():
        return None, (
            f"Model file not found at `{path}`. Run the notebook from top to "
            "bottom and place `heartsaver_model.pkl` beside `app.py`."
        )

    try:
        candidate = joblib.load(path)
    except Exception as exc:
        return None, (
            f"The model file could not be loaded: `{exc}`. Recreate it with the "
            "notebook and deploy using the same package versions."
        )

    problem = validate_bundle(candidate)
    if problem:
        return None, problem + " Re-run notebook section 6.6 to recreate the bundle."
    return candidate, None


def queue_preset(name: str) -> None:
    """Queue widget values so Streamlit applies them before creating the form."""
    st.session_state["pending_patient"] = PRESETS[name]
    st.session_state.pop("assessment", None)


def clear_form() -> None:
    """Queue an empty form and clear any result from a previous submission."""
    st.session_state["pending_patient"] = {name: None for name in PRESETS[next(iter(PRESETS))]}
    st.session_state.pop("assessment", None)


bundle, model_error = load_bundle(MODEL_PATH)
if model_error:
    st.title("HeartSaver")
    st.error(model_error)
    st.stop()

FEATURES = list(bundle["features"])
THRESHOLD = float(bundle["threshold"])
TEST_RESULTS = bundle["test_results"]

st.title("HeartSaver")
st.caption("CARDIAC REFERRAL DECISION SUPPORT")

heading_left, heading_right = st.columns([3, 2], gap="large")
with heading_left:
    st.write(
        "HeartSaver applies the final Random Forest model from the notebook to findings "
        "from a completed cardiac assessment. It indicates whether the model score "
        "reaches the selected referral threshold."
    )
with heading_right:
    st.markdown(
        f"**Model:** Random Forest  \n"
        f"**Selected inputs:** {len(FEATURES)}  \n"
        f"**Referral threshold:** {THRESHOLD:.0%}  \n"
        "**Intended stage:** after exercise stress testing"
    )

st.divider()

for field, value in st.session_state.pop("pending_patient", {}).items():
    st.session_state[f"patient_{field}"] = value

input_column, result_column = st.columns([7, 5], gap="large")

with input_column:
    st.header("Assessment inputs")
    st.write("Complete all fields using the patient's recorded assessment results.")

    example_column, load_column, clear_column = st.columns([3, 1, 1])
    selected_example = example_column.selectbox(
        "Example profile",
        list(PRESETS),
        label_visibility="collapsed",
    )
    if load_column.button("Load", use_container_width=True):
        queue_preset(selected_example)
        st.rerun()
    if clear_column.button("Clear", use_container_width=True):
        clear_form()
        st.rerun()
    st.caption("Example profiles demonstrate the interface and are not clinical recommendations.")

    with st.form("patient_assessment", border=False):
        left, right = st.columns(2, gap="large")

        with left:
            age = st.number_input(
                "Age (years)",
                min_value=18,
                max_value=100,
                value=None,
                step=1,
                placeholder="Enter age",
                key="patient_Age",
            )
            sex = st.selectbox(
                "Sex recorded in the dataset",
                list(LABELS["Sex"]),
                index=None,
                format_func=lambda value: LABELS["Sex"][value],
                placeholder="Select one",
                key="patient_Sex",
                help="The source dataset records only F and M categories.",
            )
            chest_pain = st.selectbox(
                "Chest-pain presentation",
                list(LABELS["ChestPainType"]),
                index=None,
                format_func=lambda value: LABELS["ChestPainType"][value],
                placeholder="Select one",
                key="patient_ChestPainType",
            )
            fasting_bs = st.selectbox(
                "Fasting blood sugar",
                list(LABELS["FastingBS"]),
                index=None,
                format_func=lambda value: LABELS["FastingBS"][value],
                placeholder="Select one",
                key="patient_FastingBS",
                help="Whether fasting blood sugar exceeds 120 mg/dL.",
            )

        with right:
            max_hr = st.number_input(
                "Maximum heart rate achieved (bpm)",
                min_value=30,
                max_value=240,
                value=None,
                step=1,
                placeholder="Enter maximum heart rate",
                key="patient_MaxHR",
            )
            exercise_angina = st.selectbox(
                "Exercise-induced angina",
                list(LABELS["ExerciseAngina"]),
                index=None,
                format_func=lambda value: LABELS["ExerciseAngina"][value],
                placeholder="Select one",
                key="patient_ExerciseAngina",
            )
            oldpeak = st.number_input(
                "ST depression relative to rest (Oldpeak)",
                min_value=-5.0,
                max_value=10.0,
                value=None,
                step=0.1,
                format="%.1f",
                placeholder="Enter Oldpeak",
                key="patient_Oldpeak",
                help="Negative values are accepted because they occur in the training data.",
            )
            st_slope = st.selectbox(
                "Slope of the peak exercise ST segment",
                list(LABELS["ST_Slope"]),
                index=None,
                format_func=lambda value: LABELS["ST_Slope"][value],
                placeholder="Select one",
                key="patient_ST_Slope",
            )

        submitted = st.form_submit_button(
            "Calculate model result", type="primary", use_container_width=True
        )

    if submitted:
        values = {
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain,
            "Oldpeak": oldpeak,
            "ExerciseAngina": exercise_angina,
            "MaxHR": max_hr,
            "FastingBS": fasting_bs,
            "ST_Slope": st_slope,
        }
        missing_labels = {
            "Age": "Age",
            "Sex": "Sex",
            "ChestPainType": "Chest-pain presentation",
            "Oldpeak": "ST depression (Oldpeak)",
            "ExerciseAngina": "Exercise-induced angina",
            "MaxHR": "Maximum heart rate",
            "FastingBS": "Fasting blood sugar",
            "ST_Slope": "ST-segment slope",
        }
        missing = [missing_labels[name] for name, value in values.items() if value is None]

        if missing:
            st.session_state.pop("assessment", None)
            st.error(
                "Complete every field before assessment:\n\n"
                + "\n".join(f"- {label}" for label in missing)
            )
        else:
            try:
                probability = disease_probability(bundle, values)
            except Exception as exc:
                st.session_state.pop("assessment", None)
                st.error(
                    f"The model could not score these inputs: `{exc}`. Check the deployed "
                    "model file and package versions."
                )
            else:
                st.session_state["assessment"] = {
                    "values": values,
                    "probability": probability,
                    "warnings": range_warnings(values),
                }

with result_column:
    st.header("Model result")
    result = st.session_state.get("assessment")

    if result is None:
        st.write("No result has been calculated.")
        st.write(
            "The form uses eight recorded inputs. `HasSTDepression`, the ninth model "
            "feature, is calculated automatically from Oldpeak."
        )
    else:
        probability = result["probability"]
        flagged = probability >= THRESHOLD
        decision = "Further clinical review" if flagged else "Below referral threshold"
        margin = probability - THRESHOLD

        st.markdown(f"# {probability:.1%}")
        st.markdown(f"**{decision}**")
        st.progress(
            max(0.0, min(probability, 1.0)),
            text=f"Model score {probability:.1%}; decision threshold {THRESHOLD:.0%}",
        )
        st.markdown(
            f"**Threshold comparison:** {abs(margin) * 100:.1f} percentage points "
            f"{'above' if margin >= 0 else 'below'} the threshold."
        )

        for warning in result["warnings"]:
            st.warning(warning)

        st.write(
            "This result is a statistical model output from a referred cohort. It is "
            "not a diagnosis or a calibrated estimate for the general population."
        )
        st.write(
            "A below-threshold result must not be used to dismiss symptoms or replace "
            "clinical judgement."
        )

        submitted_values = result["values"]
        st.markdown("#### Submitted values")
        st.markdown(
            f"**Age:** {submitted_values['Age']} years  \n"
            f"**Sex:** {LABELS['Sex'][submitted_values['Sex']]}  \n"
            f"**Chest-pain presentation:** "
            f"{LABELS['ChestPainType'][submitted_values['ChestPainType']]}  \n"
            f"**Fasting blood sugar:** "
            f"{LABELS['FastingBS'][submitted_values['FastingBS']]}  \n"
            f"**Maximum heart rate:** {submitted_values['MaxHR']} bpm  \n"
            f"**Exercise-induced angina:** "
            f"{LABELS['ExerciseAngina'][submitted_values['ExerciseAngina']]}  \n"
            f"**Oldpeak:** {submitted_values['Oldpeak']:.1f}  \n"
            f"**ST-segment slope:** {LABELS['ST_Slope'][submitted_values['ST_Slope']]}  \n"
            f"**Has ST depression:** "
            f"{'Yes' if submitted_values['Oldpeak'] > 0 else 'No'} (derived)"
        )

st.divider()
st.header("Evidence, scope and limitations")

evidence_column, scope_column, limitation_column = st.columns(3, gap="large")

with evidence_column:
    st.subheader("Held-out test results")
    st.markdown(
        f"| Measure | Result |\n"
        f"|---|---:|\n"
        f"| Recall | {TEST_RESULTS['recall']:.1%} |\n"
        f"| Precision | {TEST_RESULTS['precision']:.1%} |\n"
        f"| Accuracy | {TEST_RESULTS['accuracy']:.1%} |\n"
        f"| Missed cases | {TEST_RESULTS['false_negatives']} |\n"
        f"| False alarms | {TEST_RESULTS['false_positives']} |"
    )
    st.caption(
        f"Results are from {TEST_RESULTS['n_patients']} test patients at the "
        f"{THRESHOLD:.0%} threshold, read directly from the trained model file."
    )

with scope_column:
    st.subheader("Intended use")
    st.write(
        "For a clinician reviewing a completed cardiac work-up that includes exercise "
        "stress-test findings."
    )
    st.markdown(
        "**Not intended for:**  \n"
        "Self-diagnosis  \n"
        "Emergency decisions  \n"
        "Deciding whether to order a stress test"
    )
    st.write(
        f"The {THRESHOLD:.0%} threshold prioritises recall under the stated assumption "
        "that one missed case costs five false alarms."
    )

with limitation_column:
    st.subheader("Limitations")
    st.markdown(
        "- The 5:1 cost ratio was assumed rather than obtained from stakeholders.\n"
        "- The referred training cohort had 55.3% disease prevalence.\n"
        "- The threshold procedure was out-of-fold but not fully nested.\n"
        "- Earlier test analysis informed later development.\n"
        "- Subgroup findings were exploratory and based on small counts."
    )