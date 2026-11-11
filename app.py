import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="HouseAI", layout="wide")

# ==========================
# HEADER
# ==========================
st.markdown("""
# 🏠 HouseAI - Smart Price Prediction
### AI-powered real estate valuation platform
""")

# ==========================
# DATA UPLOAD
# ==========================
st.sidebar.header("📤 Upload Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("Custom dataset loaded!")
else:
    try:
        data = pd.read_csv("housing.csv")
    except:
        data = pd.read_csv("housing_dataset(1).csv")

# ==========================
# CLEAN DATA
# ==========================
data = data.dropna()

if "Address" in data.columns:
    data = data.drop("Address", axis=1)

# ==========================
# LOAD MODEL
# ==========================
model = joblib.load("lgbm_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==========================
# MAP INPUT
# ==========================
st.sidebar.subheader("📍 Select Location")

m = folium.Map(location=[19.0760, 72.8777], zoom_start=10)

map_data = st_folium(m, width=300, height=200)

lat, lon = 19.07, 72.87

if map_data and map_data["last_clicked"]:
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

st.sidebar.write(f"Lat: {lat:.4f}, Lon: {lon:.4f}")

# ==========================
# INPUT FIELDS
# ==========================
st.sidebar.header("🏡 Property Details")

features = data.drop("Price", axis=1).columns
user_input = {}

for col in features:
    if data[col].dtype in ["int64", "float64"]:
        user_input[col] = st.sidebar.slider(
            col,
            float(data[col].min()),
            float(data[col].max()),
            float(data[col].median())
        )
    else:
        user_input[col] = st.sidebar.selectbox(col, data[col].unique())

input_df = pd.DataFrame([user_input])

# Encode
for col in input_df.columns:
    if input_df[col].dtype == "object":
        input_df[col] = input_df[col].astype("category").cat.codes

# ==========================
# PREDICTION
# ==========================
input_scaled = scaler.transform(input_df)
prediction = np.expm1(model.predict(input_scaled)[0])

# ==========================
# DISPLAY RESULT
# ==========================
st.metric("💰 Estimated Price", f"₹{prediction:,.0f}")

# ==========================
# LIVE MODEL ACCURACY
# ==========================
st.subheader("📊 Model Performance")

X = data.drop("Price", axis=1)
y = data["Price"]

if uploaded_file:
    # A custom upload was not used for training, so every row is unseen data.
    X_eval, y_eval, eval_label = X, y, "uploaded data"
else:
    # Same 80/20 split as train_model.py, so only rows the model never saw are scored.
    _, X_eval, _, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)
    eval_label = "held-out 20% test split"

preds = np.expm1(model.predict(scaler.transform(X_eval)))

r2 = r2_score(y_eval, preds)
mae = mean_absolute_error(y_eval, preds)

st.caption(f"Scored on the {eval_label}, in original price units.")
col1, col2 = st.columns(2)
col1.metric("R² Score", f"{r2:.4f}")
col2.metric("MAE", f"{mae:,.0f}")

# ==========================
# VISUALS
# ==========================
st.subheader("📈 Data Visualization")

fig = px.histogram(data, x="Price", nbins=50, title="Price Distribution")
st.plotly_chart(fig, use_container_width=True)

fig2 = px.scatter(data, x=data.columns[0], y="Price", title="Feature vs Price")
st.plotly_chart(fig2, use_container_width=True)

# ==========================
# DATA VIEW
# ==========================
with st.expander("📂 View Dataset"):
    st.dataframe(data.head())