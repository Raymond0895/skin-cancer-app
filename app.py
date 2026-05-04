import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("full_pipeline.pkl")

# UI
st.title("🧬 Skin Cancer Detection")

age = st.number_input("Tuổi", 0, 100, 30)
sex = st.selectbox("Giới tính", ["male", "female"])
loc = st.selectbox("Vị trí", ["back", "lower extremity", "upper extremity"])

# Encode
sex_map = {"male": 1, "female": 0}
loc_map = {"back": 0, "lower extremity": 1, "upper extremity": 2}

# Predict
if st.button("Dự đoán"):
    input_data = np.array([[age, sex_map[sex], loc_map[loc]]])

    # ❌ BỎ scaler đi
    pred = model.predict(input_data)[0]

    if pred == 1:
        st.error("⚠️ Nguy cơ ung thư")
    else:
        st.success("✅ Lành tính")
