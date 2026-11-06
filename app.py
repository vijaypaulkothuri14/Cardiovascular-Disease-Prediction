
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
MODEL_PATH = OUTPUT / "best_cardio_model.joblib"
RESULT_PATH = OUTPUT / "model_comparison.csv"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CardioCare | Heart Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: #f3f7fb;
}
[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}
.hero {
    background: linear-gradient(115deg, #102d46, #087e8b);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 35px;
    margin-bottom: 8px;
}
.hero p {
    color: #e0f5f5;
    font-size: 16px;
}
.section-card {
    background: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #e3ebf1;
    margin-bottom: 18px;
}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e1eaf0;
    text-align: center;
}
.metric-label {
    color: #637789;
    font-size: 14px;
}
.metric-value {
    color: #087e8b;
    font-size: 27px;
    font-weight: 700;
}
.result-positive {
    background: #fff0ee;
    border: 1px solid #f2b6ae;
    padding: 22px;
    border-radius: 14px;
    color: #9b3028;
}
.result-negative {
    background: #e9f8f1;
    border: 1px solid #a7dec2;
    padding: 22px;
    border-radius: 14px;
    color: #176c48;
}
div.stButton > button[kind="primary"] {
    background: #087e8b;
    color: white;
    border: none;
    border-radius: 9px;
    height: 48px;
    font-weight: 600;
}
div.stButton > button[kind="primary"]:hover {
    background: #075f69;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_saved_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

saved = load_saved_model()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ❤️ CardioCare")
    st.caption("Machine Learning Project")
    st.divider()

    st.markdown("### Project information")
    st.write("**Project:** Cardiovascular Disease Prediction")
    st.write("**Algorithm:** Random Forest")
    st.write("**Dataset:** Cardiovascular health records")

    st.divider()
    st.markdown("### Model performance")

    if RESULT_PATH.exists():
        results = pd.read_csv(RESULT_PATH)
        best = results.sort_values(
            "Accuracy", ascending=False
        ).iloc[0]

        st.metric(
            "Best test accuracy",
            f"{best['Accuracy']:.2%}"
        )
        st.caption(f"Best algorithm: {best['Model']}")

    st.divider()
    st.caption("Developed using Python, Streamlit and Scikit-learn.")

# ---------------- MAIN HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>❤️ Cardiovascular Health Prediction</h1>
    <p>
        An intelligent machine learning application
        for educational cardiovascular disease prediction.
    </p>
</div>
""", unsafe_allow_html=True)

if saved is None:
    st.error(
        "Trained model not found. Run python main.py first."
    )
    st.stop()

model = saved["model"]
features = saved["features"]

# ---------------- TOP METRICS ----------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Prediction Model</div>
        <div class="metric-value">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Dataset Records</div>
        <div class="metric-value">70,000</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    accuracy_text = "Not available"
    if RESULT_PATH.exists():
        results = pd.read_csv(RESULT_PATH)
        best = results.sort_values(
            "Accuracy", ascending=False
        ).iloc[0]
        accuracy_text = f"{best['Accuracy']:.2%}"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Best Test Accuracy</div>
        <div class="metric-value">{accuracy_text}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------- PATIENT FORM ----------------
st.markdown("## Patient Health Information")
st.write("Enter the available health measurements below.")

with st.form("health_form"):

    left, right = st.columns(2)

    with left:
        st.markdown("### Personal information")

        age = st.number_input(
            "Age (years)", 18, 100, 50
        )

        gender = st.selectbox(
            "Gender code (dataset)",
            [1, 2],
            format_func=lambda x: f"Code {x}"
        )

        height = st.number_input(
            "Height (cm)", 100, 250, 165
        )

        weight = st.number_input(
            "Weight (kg)", 30.0, 250.0, 70.0
        )

        st.markdown("### Blood pressure")

        ap_hi = st.number_input(
            "Systolic pressure", 70, 250, 120
        )

        ap_lo = st.number_input(
            "Diastolic pressure", 40, 150, 80
        )

    with right:
        st.markdown("### Health measurements")

        cholesterol = st.selectbox(
            "Cholesterol",
            [1, 2, 3],
            format_func=lambda x: {
                1: "1 - Normal",
                2: "2 - Above normal",
                3: "3 - Well above normal"
            }[x]
        )

        gluc = st.selectbox(
            "Glucose",
            [1, 2, 3],
            format_func=lambda x: {
                1: "1 - Normal",
                2: "2 - Above normal",
                3: "3 - Well above normal"
            }[x]
        )

        smoke = st.selectbox(
            "Smoking",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        alco = st.selectbox(
            "Alcohol consumption",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        active = st.selectbox(
            "Physical activity",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    submitted = st.form_submit_button(
        "❤️  Predict Cardiovascular Disease",
        type="primary",
        use_container_width=True
    )

# ---------------- PREDICTION ----------------
if submitted:

    if ap_hi <= ap_lo:
        st.error(
            "Please check the blood pressure values. "
            "Systolic pressure must be greater than diastolic pressure."
        )

    else:
        patient = {
            "age": float(age),
            "gender": int(gender),
            "height": int(height),
            "weight": float(weight),
            "ap_hi": int(ap_hi),
            "ap_lo": int(ap_lo),
            "cholesterol": int(cholesterol),
            "gluc": int(gluc),
            "smoke": int(smoke),
            "alco": int(alco),
            "active": int(active)
        }

        input_data = pd.DataFrame(
            [patient], columns=features
        )

        prediction = int(model.predict(input_data)[0])

        st.markdown("---")
        st.markdown("## Prediction Result")

        if prediction == 1:
            st.markdown("""
            <div class="result-positive">
                <h3>Model prediction: Class 1</h3>
                <p>
                    The model classified this input
                    into the cardiovascular disease category.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="result-negative">
                <h3>Model prediction: Class 0</h3>
                <p>
                    The model classified this input
                    into the no cardiovascular disease category.
                </p>
            </div>
            """, unsafe_allow_html=True)

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(
                input_data
            )[0][1]

            st.metric(
                "Model-estimated probability for class 1",
                f"{probability:.2%}"
            )

            st.caption(
                "This probability is a model output, not a "
                "clinically validated assessment of personal risk."
            )

        with st.expander("View entered patient information"):
            st.dataframe(
                input_data,
                use_container_width=True
            )

# ---------------- MODEL COMPARISON ----------------
st.markdown("---")
st.markdown("## Machine Learning Model Comparison")

if RESULT_PATH.exists():
    results = pd.read_csv(RESULT_PATH)
    results = results.sort_values(
        "Accuracy", ascending=False
    )

    st.bar_chart(
        results.set_index("Model")["Accuracy"]
    )

    st.dataframe(
        results.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}",
            "ROC AUC": "{:.2%}"
        }),
        use_container_width=True
    )

# ---------------- FOOTER ----------------
st.markdown("---")

st.info(
    "Medical disclaimer: This is an educational machine learning "
    "demonstration, not a medical diagnostic system. It must not "
    "be used to make healthcare decisions. Consult a qualified "
    "healthcare professional for medical advice."
)

st.caption(
    "CardioCare | Cardiovascular Disease Prediction | "
    "Developed using Python and Machine Learning"
)
