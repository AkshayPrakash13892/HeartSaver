"""HeartSaver — Streamlit interface for the model built in HeartSaver.ipynb.

The notebook writes ``heartsaver_model.pkl`` as a four-key dictionary containing
the fitted pipeline, selected feature list, out-of-fold decision threshold, and the
test-set results that threshold produced. This app reads that complete contract so
the model inputs, the 0.30 operating point, and the displayed metrics cannot
silently drift apart from whatever the notebook last produced.
"""

import base64
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="HeartSaver — cardiac referral support",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(__file__).parent / "heartsaver_model.pkl"
BACKGROUND_IMAGE_PATH = Path(__file__).parent / "background_for_app.png"

# The 9 features the deployed model was actually trained on (notebook 6.2).
# Used to sanity-check a loaded .pkl against what this form collects.
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

# Ranges seen in training. Inputs outside these are still accepted (they can
# be clinically real) but get flagged as extrapolation.
TRAINING_RANGES = {
    "Age": (28, 77),
    "MaxHR": (60, 202),
    "Oldpeak": (-2.6, 6.2),
}

# Category codes -> display labels for every dropdown.
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

# Three canned patients for the "Load" button — first two match the
# notebook's own worked examples, third sits near the 0.30 cutoff.
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


# ---------------------------------------------------------------------------
# Look and feel
#
# Palette now follows the background image itself instead of fighting it:
# near-black navy surfaces, light grey-white text, and a single red accent
# lifted straight from the ECG trace in the picture. Blue is kept only as
# the secondary/"calm" colour (it's the glow around the heart in the image)
# so the two decision states read the way the picture already reads —
# blue and steady vs. red and flagged.
# ---------------------------------------------------------------------------

def _background_css() -> str:
    """Build the CSS for the page background from background_for_app.png.

    Streamlit doesn't serve arbitrary files from the app folder over HTTP,
    so a local image can't just be linked with a normal url("./file.png") —
    the browser has no route to it. Inlining it as base64 straight into the
    stylesheet sidesteps that. The scrim on top is a dark navy wash now
    (matching the image's own tone) rather than a light paper wash, so it
    darkens for legibility without turning the picture pale. Falls back to
    a flat dark colour if the file isn't there yet, rather than breaking
    the whole page over a missing asset.
    """
    if not BACKGROUND_IMAGE_PATH.exists():
        return "background-color: var(--bg);"

    encoded = base64.b64encode(BACKGROUND_IMAGE_PATH.read_bytes()).decode("utf-8")
    return (
        "background-color: var(--bg);"
        "background-image: linear-gradient(rgba(8, 14, 24, 0.35), "
        f'rgba(8, 14, 24, 0.35)), url("data:image/png;base64,{encoded}");'
        "background-size: cover, cover;"
        "background-position: center, center;"
        "background-attachment: fixed, fixed;"
        "background-repeat: no-repeat, no-repeat;"
    )


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --bg:       #0A1420;   /* page background, matches the image's navy */
    --text:     #E7EDF3;   /* body text / headings on dark surfaces */
    --line:     #3C5872;   /* borders, dividers on dark surfaces */
    --red:      #FF3B3B;   /* ECG-trace red — flagged / primary accent */
    --blue:     #4FA8E8;   /* heart-glow blue — calm / secondary accent */
    --surface:  #101B29;   /* solid fallback card colour */
}

[data-testid="stAppViewContainer"] { __BACKGROUND_CSS__ }
[data-testid="stHeader"] { background: transparent; }

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; color: var(--text); }

/* Streamlit's own containers (st.container(border=True)) become the card
   surfaces. Translucent dark + blurred rather than solid, so the
   background image reads through every panel instead of just the gaps
   between them — the blur is what keeps text legible over the busier
   parts of the picture (the heart, the trace). */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(10, 18, 30, 0.55);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border: 1px solid rgba(60, 88, 114, 0.6) !important;
    border-radius: 10px;
}

h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; letter-spacing: -0.01em; color: var(--text); }
p, li, label, span { color: var(--text); }
[data-testid="stCaptionContainer"] { color: rgba(231, 237, 243, 0.65); }

/* Numbers read like a monitor: monospace, no ambiguity between similar digits. */
.vital-readout {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    padding: 0.9rem 1.1rem;
    border-radius: 8px;
    border-left: 5px solid var(--line);
    background: rgba(8, 14, 24, 0.55);
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
    margin-bottom: 0.6rem;
}
.vital-readout .vital-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.6rem;
    font-weight: 600;
    line-height: 1;
}
.vital-readout .vital-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
    font-size: 0.95rem;
    color: rgba(231, 237, 243, 0.8);
}
/* Below threshold reads as the calm blue glow; at/above threshold as the
   ECG-spike red — the same two colours already doing that job in the image. */
.readout-clear { border-left-color: var(--blue); }
.readout-clear .vital-value { color: var(--blue); }
.readout-refer { border-left-color: var(--red); }
.readout-refer .vital-value { color: var(--red); }

.masthead-mark {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 1.7rem;
    color: var(--text);
}
.masthead-mark svg { flex-shrink: 0; }
.masthead-kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    color: var(--red);
    text-transform: uppercase;
}

/* --- Buttons ---------------------------------------------------------- */
/* Secondary buttons (Load / Clear): dark translucent, light text, red on
   hover — same idea as before, just recoloured for the dark theme. */
.stButton > button {
    background: rgba(16, 27, 41, 0.6);
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
    color: var(--text);
    border: 1px solid rgba(60, 88, 114, 0.8);
    border-radius: 6px;
    font-weight: 500;
    transition: border-color 0.15s ease, color 0.15s ease, background 0.15s ease;
}
.stButton > button:hover {
    border-color: var(--red);
    color: var(--red);
    background: rgba(16, 27, 41, 0.8);
}
/* Primary button (Calculate model result): solid red fill, matching the
   ECG trace — Streamlit renders this with a kind="primary" attribute. */
.stButton > button[kind="primary"],
button[kind="primaryFormSubmit"] {
    background: rgba(255, 59, 59, 0.88);
    border-color: rgba(255, 59, 59, 0.88);
    color: #FFFFFF !important;
}
.stButton > button[kind="primary"]:hover,
button[kind="primaryFormSubmit"]:hover {
    background: #FF3B3B;
    border-color: #FF3B3B;
    color: #FFFFFF !important;
}

/* --- Inputs ------------------------------------------------------------ */
/* Number inputs and the selectbox trigger, restyled for the dark theme —
   dark translucent fields with light text instead of Streamlit's default
   white-on-white, which would disappear against this background. */
[data-testid="stNumberInput"] input {
    background: rgba(16, 27, 41, 0.65);
    color: var(--text);
    border-color: rgba(60, 88, 114, 0.8) !important;
}
[data-baseweb="select"] > div {
    background: rgba(16, 27, 41, 0.65) !important;
    border-color: rgba(60, 88, 114, 0.8) !important;
    color: var(--text);
}
[data-baseweb="select"] input { color: var(--text) !important; }
/* The dropdown menu renders in a floating layer outside the form, so it
   needs its own rule to pick up the same palette. Kept close to opaque
   since it floats directly over the image when open. */
[data-baseweb="popover"] [data-baseweb="menu"] {
    background: rgba(10, 18, 30, 0.95);
}
[data-baseweb="menu"] li { color: var(--text); }
[data-baseweb="menu"] li:hover { background: rgba(60, 88, 114, 0.4); }

/* Focus state uses the red accent, matching the rest of the theme. */
[data-baseweb="select"]:focus-within > div,
[data-testid="stNumberInput"] input:focus {
    border-color: var(--red) !important;
    box-shadow: 0 0 0 1px var(--red) !important;
}

/* Progress bar fill (the score bar under the readout) — red, same accent
   as the ECG trace and the primary button. */
[data-testid="stProgress"] > div > div > div {
    background-color: var(--red) !important;
}

[data-testid="stMarkdownContainer"] table {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.92rem;
    color: var(--text);
}
[data-testid="stMarkdownContainer"] table th,
[data-testid="stMarkdownContainer"] table td {
    border-color: rgba(60, 88, 114, 0.5) !important;
}

/* Streamlit's default alert boxes (st.error / st.warning) keep their own
   colouring for clarity, but need a dark-theme background so they don't
   render as a pale box on this palette. */
[data-testid="stAlert"] { background: rgba(16, 27, 41, 0.75); }
</style>
""".replace("__BACKGROUND_CSS__", _background_css())

st.markdown(CSS, unsafe_allow_html=True)


def validate_bundle(candidate: Any) -> str | None:
    """Check that whatever we just unpickled is actually a usable model
    bundle, and say specifically what's wrong if it isn't. Better to fail
    here with a readable message than a KeyError three functions later.
    """
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

    # The pkl's feature list has to line up with what the form below
    # actually collects, or we'd be asking the user for the wrong things.
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
    """Turn one filled-out form into a single model-ready row.

    HasSTDepression isn't a form field — the model expects it as a column
    but we only ask for Oldpeak, so it's derived here the same way the
    notebook derives it in 3.2.
    """
    row = dict(values)
    row["HasSTDepression"] = int(float(row["Oldpeak"]) > 0)
    return pd.DataFrame([row]).loc[:, features]  # reorder/subset to match the fitted pipeline


def disease_probability(bundle: dict[str, Any], values: dict[str, Any]) -> float:
    """Score one patient and return P(heart disease)."""
    model = bundle["model"]
    features = list(bundle["features"])
    # classes_ isn't guaranteed to be ordered [0, 1], so look up the
    # positive class's column instead of assuming index 1.
    positive_column = list(model.classes_).index(1)
    probability = model.predict_proba(patient_frame(values, features))[0, positive_column]
    return float(probability)


def range_warnings(values: dict[str, Any]) -> list[str]:
    """Flag any numeric input that falls outside what the model was
    actually trained on, so a wildly out-of-range prediction doesn't
    look as confident as one grounded in the data."""
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
    """Load and validate the .pkl once per session. Returns (bundle, None)
    on success or (None, message) on failure — never raises, so the caller
    can just check model_error and st.stop() cleanly."""
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
    """Stage a preset so it lands in the form widgets on the next rerun,
    and wipe any result left over from before."""
    st.session_state["pending_patient"] = PRESETS[name]
    st.session_state.pop("assessment", None)

def clear_form() -> None:
    """Same idea as queue_preset, but every field goes back to empty."""
    st.session_state["pending_patient"] = {name: None for name in PRESETS[next(iter(PRESETS))]}
    st.session_state.pop("assessment", None)


# If the model can't be loaded there's nothing else worth rendering.
bundle, model_error = load_bundle(MODEL_PATH)
if model_error:
    st.title("HeartSaver")
    st.error(model_error)
    st.stop()

FEATURES = list(bundle["features"])
THRESHOLD = float(bundle["threshold"])
TEST_RESULTS = bundle["test_results"]

with st.container(border=True):
    heading_left, heading_right = st.columns([3, 2], gap="large")
    with heading_left:
        # Small inline heartbeat mark next to the wordmark, stroked in the
        # same red as the ECG trace in the background image.
        st.markdown(
            f"""
            <div class="masthead-kicker">Cardiac referral support</div>
            <div class="masthead-mark">
                <svg width="34" height="20" viewBox="0 0 110 30" xmlns="http://www.w3.org/2000/svg">
                    <path d="M0,15 L17,15 L20,11 L23,19 L26,4 L29,22 L32,15 L44,15 L47,12 L50,15 L110,15"
                          fill="none" stroke="#FF3B3B" stroke-width="2.4"
                          stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                HeartSaver
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(
            "Applies the final Random Forest model from the notebook to findings "
            "from a completed cardiac assessment. Indicates whether the model score "
            "reaches the selected referral threshold."
        )
    with heading_right:
        st.markdown(
            f"**Model:** Random Forest  \n"
            f"**Selected inputs:** {len(FEATURES)}  \n"
            f"**Referral threshold:** {THRESHOLD:.0%}  \n"
            "**Intended stage:** after exercise stress testing"
        )

st.write("")  # a little breathing room before the working area

# Any preset/clear click from the last rerun gets applied here, before the
# widgets below are created, so Streamlit picks up the new values.
for field, value in st.session_state.pop("pending_patient", {}).items():
    st.session_state[f"patient_{field}"] = value

input_column, result_column = st.columns([7, 5], gap="large")

with input_column:
    with st.container(border=True):
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
            st.rerun()  # need a rerun so the staged values reach the widgets
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
                    value=None,  # start blank so a submit with nothing entered is caught below
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
                # Don't score a half-filled form — clear any stale result and
                # tell the user exactly what's left.
                st.session_state.pop("assessment", None)
                st.error(
                    "Complete every field before assessment:\n\n"
                    + "\n".join(f"- {label}" for label in missing)
                )
            else:
                try:
                    probability = disease_probability(bundle, values)
                except Exception as exc:
                    # Shouldn't happen if validate_bundle passed, but if the
                    # pipeline chokes for some other reason, say so instead
                    # of showing a traceback.
                    st.session_state.pop("assessment", None)
                    st.error(
                        f"The model could not score these inputs: `{exc}`. Check the deployed "
                        "model file and package versions."
                    )
                else:
                    # Stash the result so it survives the rerun and shows up
                    # in the result column on the right.
                    st.session_state["assessment"] = {
                        "values": values,
                        "probability": probability,
                        "warnings": range_warnings(values),
                    }

with result_column:
    with st.container(border=True):
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

            # Styled as a monitor readout: colour carries the decision so
            # it's legible at a glance, not just from the printed label.
            badge_class = "readout-refer" if flagged else "readout-clear"
            st.markdown(
                f"""
                <div class="vital-readout {badge_class}">
                    <span class="vital-value">{probability:.1%}</span>
                    <span class="vital-label">{decision}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(
                max(0.0, min(probability, 1.0)),  # clamp, just in case
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

            # Echo the inputs back, including the derived feature, so the user
            # can check what actually produced this score.
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

st.write("")
st.header("Evidence, scope and limitations")

evidence_column, scope_column, limitation_column = st.columns(3, gap="large")

with evidence_column:
    with st.container(border=True):
        # Pulled straight from the bundle, never recomputed here, so these
        # numbers can't drift from what the notebook actually measured.
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
    with st.container(border=True):
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
    with st.container(border=True):
        st.subheader("Limitations")
        st.markdown(
            "- The 5:1 cost ratio was assumed rather than obtained from stakeholders.\n"
            "- The referred training cohort had 55.3% disease prevalence.\n"
            "- The threshold procedure was out-of-fold but not fully nested.\n"
            "- Earlier test analysis informed later development.\n"
            "- Subgroup findings were exploratory and based on small counts."
        )