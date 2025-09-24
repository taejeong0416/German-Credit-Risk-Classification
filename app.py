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

model = load_model()

st.title("Credit Risk Prediction")

# --- 진단: 모델이 기대하는 입력 형태/컬럼 출력 ---
st.subheader("Model input schema (diagnostic)")
try:
    if hasattr(model, "feature_names_in_"):
        st.write("feature_names_in_ (len):", len(model.feature_names_in_))
        st.write(list(model.feature_names_in_))
    elif isinstance(model, Pipeline):
        st.write("Pipeline steps:", list(model.named_steps.keys()))
        last = list(model.named_steps.values())[-1]
        if hasattr(last, "feature_names_in_"):
            st.write("final estimator feature_names_in_ (len):", len(last.feature_names_in_))
            st.write(list(last.feature_names_in_))
        else:
            st.info("final estimator has no feature_names_in_.")
    else:
        st.info("No feature_names_in_. Likely trained on a numpy array.")
except Exception as e:
    st.error("Schema inspection failed")
    st.exception(e)

# --- 기존 UI ---
age = st.number_input("Age", 18, 100, 30)
amount = st.number_input("Credit amount", 100, 50000, 2000, 100)
duration = st.number_input("Duration (months)", 4, 72, 24)
sex = st.selectbox("Sex", ["male", "female"])
job = st.selectbox("Job", ["0","1","2","3"])
housing = st.selectbox("Housing", ["own", "free", "rent"])

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Credit amount": amount,
        "Duration": duration,
        "Sex": sex,
        "Job": job,
        "Housing": housing
    }])

    # 실행 전 최종 입력/예상 입력 비교 표시
    st.write({"input_columns": list(input_df.columns), "n_cols": input_df.shape[1]})

    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else float("nan")
        st.write(f"Predicted class: {pred}")
        st.write(f"Probability of positive class: {proba:.3f}")
    except Exception as e:
        st.error("Prediction failed:")
        st.exception(e)
