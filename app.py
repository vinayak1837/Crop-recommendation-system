import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

# ----------------------------
# Load Model, Scaler & Encoder
# ----------------------------
model = pickle.load(open("model.pkl", "rb"))
mx = pickle.load(open("minmaxscalar.pkl", "rb"))      # Use if trained with MinMaxScaler
le = pickle.load(open("label_encoder.pkl", "rb"))    # Label Encoder (VERY IMPORTANT)

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="🌾 Smart Crop Recommendation", layout="wide")

st.title("🌾 Smart Crop Recommendation System")
st.markdown("AI-based Crop Prediction using Soil & Climate Data")

st.write("---")

# ----------------------------
# Crop Description Dictionary
# ----------------------------
crop_info = {
    "rice": "🌾 Rice grows best in high rainfall and warm temperatures.",
    "maize": "🌽 Maize requires moderate rainfall and well-drained soil.",
    "chickpea": "🌱 Chickpea grows well in dry and cool climate.",
    "kidneybeans": "🫘 Kidney beans prefer moderate rainfall and slightly acidic soil.",
    "pigeonpeas": "🌿 Pigeon peas are drought resistant crops.",
    "mothbeans": "🌿 Moth beans grow well in dry regions.",
    "mungbean": "🌱 Mung beans require warm temperature and moderate rainfall.",
    "blackgram": "🌾 Black gram grows best in warm and humid climate.",
    "lentil": "🌾 Lentils prefer cool weather and low rainfall.",
    "pomegranate": "🍎 Pomegranate grows in dry climate with low rainfall.",
    "banana": "🍌 Banana requires high humidity and heavy rainfall.",
    "mango": "🥭 Mango grows well in tropical warm climate.",
    "grapes": "🍇 Grapes require warm and dry climate.",
    "watermelon": "🍉 Watermelon prefers hot climate and sandy soil.",
    "muskmelon": "🍈 Muskmelon grows in warm climate with low humidity.",
    "apple": "🍎 Apple requires cool climate.",
    "orange": "🍊 Orange grows best in subtropical climate.",
    "papaya": "🍈 Papaya grows well in warm tropical climate.",
    "coconut": "🥥 Coconut requires high humidity and rainfall.",
    "cotton": "🌿 Cotton grows best in black soil and warm climate.",
    "jute": "🌾 Jute requires high rainfall and humid climate.",
    "coffee": "☕ Coffee grows in cool and humid climate."
}

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("🧪 Enter Soil & Climate Details")


N = st.sidebar.slider("Nitrogen (N)", 0, 140, 50)
P = st.sidebar.slider("Phosphorus (P)", 5, 145, 50)
K = st.sidebar.slider("Potassium (K)", 5, 205, 50)

temperature = st.sidebar.slider("Temperature (°C)", 8.0, 45.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 60.0)
ph = st.sidebar.slider("pH Level", 3.5, 9.5, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 150.0)

input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

# ----------------------------
# Prediction
# ----------------------------
if st.sidebar.button("🌱 Recommend Crop"):

    # IMPORTANT: Use the SAME scaler you used during training
    input_scaled = mx.transform(input_data)  
    

    prediction = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)

    # Convert numeric prediction back to crop name
    crop_name = le.inverse_transform(prediction)[0]

    st.success(f"✅ Recommended Crop: **{crop_name.upper()}**")

    # Crop Description
    if crop_name in crop_info:
        st.info(crop_info[crop_name])
    else:
        st.write("No description available.")

    st.write("---")

    # ----------------------------
    # Probability Chart
    # ----------------------------
    prob_df = pd.DataFrame(proba, columns=le.classes_)
    prob_df = prob_df.T.reset_index()
    prob_df.columns = ["Crop", "Probability"]

    fig = px.bar(
        prob_df,
        x="Crop",
        y="Probability",
        title="Prediction Confidence",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Feature Explanation
# ----------------------------
with st.expander("📘 Feature Explanation"):
    st.write("""
    - Nitrogen (N): Essential for leaf growth  
    - Phosphorus (P): Root development  
    - Potassium (K): Disease resistance  
    - Temperature: Climate suitability  
    - Humidity: Moisture level  
    - pH: Soil acidity/alkalinity  
    - Rainfall: Water availability  
    """)

st.write("---")
st.markdown("Built with using Streamlit & Machine Learning")
