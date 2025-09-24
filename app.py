import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline


st.set_page_config(page_title="Credit Risk Prediction")

@st.cache_resource
def load_model():
    p = Path(__file__).parent / "model.pkl"
    return joblib.load(p)

def expected_input_columns(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if isinstance(model, Pipeline):
        for _, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def build_feature_row(
    age, amount, duration, sex, job, housing,
    saving, checking, purpose, monthly_payment,
    expected_cols
):
    row = {
        "Age": age,
        "Job": int(job),
        "Saving accounts": {"little":0, "moderate":1, "quite rich":2, "rich":3, "Unknown":4}[saving],
        "Credit amount": amount,
        "Monthly payment": monthly_payment,
        "Sex_male": 1 if sex == "male" else 0,
        "Housing_own": 1 if housing == "own" else 0,
        "Housing_rent": 1 if housing == "rent" else 0,
        "Checking account_little": 1 if checking == "little" else 0,
        "Checking account_moderate": 1 if checking == "moderate" else 0,
        "Checking account_rich": 1 if checking == "rich" else 0,
        "Purpose_car": 1 if purpose == "car" else 0,
        "Purpose_domestic appliances": 1 if purpose == "domestic appliances" else 0,
        "Purpose_education": 1 if purpose == "education" else 0,
        "Purpose_furniture/equipment": 1 if purpose == "furniture/equipment" else 0,
        "Purpose_radio/TV": 1 if purpose == "radio/TV" else 0,
        "Purpose_repairs": 1 if purpose == "repairs" else 0,
        "Purpose_vacation/others": 1 if purpose == "vacation/others" else 0,
    }
    if expected_cols is not None:
        for c in expected_cols:
            row.setdefault(c, 0)
        row = {k: row[k] for k in expected_cols}
    return row

model = load_model()

st.title("Credit Risk Prediction")

st.subheader("Model input schema (diagnostic)")
try:
    exp_cols = expected_input_columns(model)
    if exp_cols is not None:
        st.write("feature_names_in_ (len):", len(exp_cols))
        st.write(exp_cols)
    else:
        st.info("No feature_names_in_. Using fallback column list.")
except Exception as e:
    st.error("Schema inspection failed")
    st.exception(e)

age = st.number_input("Age", 18, 100, 30)
amount = st.number_input("Credit amount", 100, 50000, 2000, 100)
duration = st.number_input("Duration (months)", 4, 72, 24)
sex = st.selectbox("Sex", ["male", "female"])
job = st.selectbox("Job", ["0","1","2","3"])
housing = st.selectbox("Housing", ["own", "free", "rent"])
saving = st.selectbox("Saving accounts", ["little","moderate","quite rich","rich","Unknown"])
checking = st.selectbox("Checking account", ["Unknown","little","moderate","rich"])
purpose = st.selectbox(
    "Purpose",
    ["business","car","domestic appliances","education","furniture/equipment","radio/TV","repairs","vacation/others"]
)
monthly_payment = st.number_input("Monthly payment", 1, 2000, 50)

if st.button("Predict"):
    if exp_cols is None:
        exp_cols = [
            "Age","Job","Saving accounts","Credit amount","Monthly payment",
            "Sex_male","Housing_own","Housing_rent",
            "Checking account_little","Checking account_moderate","Checking account_rich",
            "Purpose_car","Purpose_domestic appliances","Purpose_education",
            "Purpose_furniture/equipment","Purpose_radio/TV","Purpose_repairs","Purpose_vacation/others"
        ]

    row = build_feature_row(
        age, amount, duration, sex, job, housing,
        saving, checking, purpose, monthly_payment,
        expected_cols=exp_cols
    )
    input_df = pd.DataFrame([row]).reindex(columns=exp_cols)

    st.write({"input_columns": list(input_df.columns), "n_cols": input_df.shape[1]})

    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else float("nan")
        st.write(f"Predicted class: {pred}")
        st.write(f"Probability of positive class: {proba:.3f}")
    except Exception as e:
        st.error("Prediction failed:")
        st.exception(e)
