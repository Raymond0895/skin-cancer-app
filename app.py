import streamlit as st
import numpy as np
import joblib

# Load model đã train

data = joblib.load("full_pipeline.pkl")

model = data["model"]
scaler = data["scaler"]

# Giao diện

st.title("🧬 Skin Cancer Detection")

age = st.number_input("Tuổi", 0, 100, 30)
sex = st.selectbox("Giới tính", ["male", "female"])
loc = st.selectbox("Vị trí", ["back", "lower extremity", "upper extremity"])

# Encode giống lúc train

sex_map = {"male": 1, "female": 0}
loc_map = {"back": 0, "lower extremity": 1, "upper extremity": 2}

# Predict

if st.button("Dự đoán"):
    input_data = np.array([[age, sex_map[sex], loc_map[loc]]])
    input_data = scaler.transform(input_data)

    pred = model.predict(input_data)[0]

    if pred == 1:
        st.error("⚠️ Nguy cơ ung thư")
    else:
        st.success("✅ Lành tính")
