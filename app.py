import streamlit as st
import pandas as pd
import joblib

# ------------------------
# 모델 불러오기
# ------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")   # 학습 단계에서 저장한 모델

model = load_model()

st.title("Credit Risk Prediction")

# ------------------------
# 입력값 받기
# ------------------------
age = st.number_input("Age", 18, 100, 30)
amount = st.number_input("Credit amount", 100, 50000, 2000, 100)
duration = st.number_input("Duration (months)", 4, 72, 24)
sex = st.selectbox("Sex", ["male", "female"])
job = st.selectbox("Job", ["0","1","2","3"])
housing = st.selectbox("Housing", ["own", "free", "rent"])

# ------------------------
# 예측 실행
# ------------------------
if st.button("Predict"):
    # 입력값 DataFrame으로 변환 (칼럼명은 학습할 때 쓴 것과 동일해야 함)
    input_df = pd.DataFrame([{
        "Age": age,
        "Credit amount": amount,
        "Duration": duration,
        "Sex": sex,
        "Job": job,
        "Housing": housing
    }])

    # 예측
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]  # class=1의 확률 (good/bad 기준에 맞게 조정)

    st.write(f"Predicted class: {pred}")
    st.write(f"Probability of positive class: {prob:.2f}")
