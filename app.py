import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Smart Water Purification",
    page_icon="💧",
    layout="wide"
)

# -----------------------------
# LOAD MODEL & SCALER (CACHED)
# -----------------------------
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        return model, scaler
    except:
        return None, None

model, scaler = load_model()

if model is None or scaler is None:
    st.error("❌ Model or scaler file not found!")
    st.stop()

# -----------------------------
# LOAD DATA (CACHED)
# -----------------------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("water_purification_dataset.csv")
    except:
        return None

df = load_data()

# -----------------------------
# UI START
# -----------------------------
st.title("💧 Smart Water Purification System")
st.markdown("### 🚰 AI-based Water Quality Prediction")

menu = st.sidebar.radio(
    "Navigation",
    ["Prediction", "Graphs", "Model Performance"]
)

# =============================
# 1. PREDICTION
# =============================
if menu == "Prediction":

    st.subheader("🔍 Enter Sensor Values")

    col1, col2 = st.columns(2)

    with col1:
        pH = st.slider("pH", 6.0, 9.0, 7.0)
        turbidity = st.slider("Turbidity (NTU)", 0.1, 10.0, 2.0)
        tds = st.slider("TDS (ppm)", 100, 1000, 300)
        flow = st.slider("Flow Rate (L/min)", 0.5, 2.0, 1.0)

    with col2:
        pressure = st.slider("Pressure (bar)", 1.0, 5.0, 2.5)
        temp = st.slider("Temperature (°C)", 15.0, 35.0, 25.0)
        usage = st.slider("Usage (L/day)", 5, 50, 20)
        days = st.slider("Days Since Filter Change", 1, 180, 60)

    if st.button("🚀 Predict", use_container_width=True):

        input_data = np.array([[pH, turbidity, tds, flow, pressure, temp, usage, days]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        st.subheader("Result")

        if prediction == 0:
            st.success("✅ Water Quality: SAFE")
        elif prediction == 1:
            st.warning("⚠️ Water Quality: MODERATE")
        else:
            st.error("❌ Water Quality: UNSAFE")

# =============================
# 2. GRAPHS
# =============================
elif menu == "Graphs":

    st.subheader("📊 Data Insights")

    if df is not None:

        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots()
            sns.countplot(x='water_quality', data=df, ax=ax1)
            ax1.set_title("Water Quality Distribution")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots()
            sns.scatterplot(
                x='TDS_ppm',
                y='turbidity_NTU',
                hue='water_quality',
                data=df,
                ax=ax2
            )
            ax2.set_title("TDS vs Turbidity")
            st.pyplot(fig2)

    else:
        st.warning("⚠️ Dataset not available")

# =============================
# 3. MODEL PERFORMANCE
# =============================
elif menu == "Model Performance":

    st.subheader("📈 Confusion Matrix")

    if df is not None:

        X = df.drop(columns=["water_quality", "filter_replacement", "maintenance_required"])
        y = df["water_quality"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        cm = confusion_matrix(y_test, y_pred)

        labels = ["Safe", "Moderate", "Unsafe"]

        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        st.pyplot(fig)

    else:
        st.warning("⚠️ Dataset not available")
