import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forest Cover Type Prediction",
    page_icon="🌲",
    layout="centered",
)

import base64
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(
                    rgba(0,0,0,0.7),
                    rgba(0,0,0,0.9)
                ),
                url("data:image/png;base64,{data}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg(r"C:\Users\sathy\Downloads\forest_bg.png")
# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("random_forest.pkl", "rb") as f:
        model = pickle.load(f)
    scaler = joblib.load("scaler.pkl")
    with open("columns.pkl", "rb") as f:
        columns = pickle.load(f)
    return model, scaler, columns

try:
    model, scaler, columns = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError as e:
    st.error(f"⚠️ Could not load model files: {e}\n\nMake sure `random_forest.pkl`, `scaler.pkl`, and `columns.pkl` are in the same folder as this script.")
    artifacts_loaded = False

# ── Label map ──────────────────────────────────────────────────────────────────
# LabelEncoder encodes alphabetically:
# 0=Aspen, 1=Cottonwood/Willow, 2=Douglas-fir,
# 3=Krummholz, 4=Lodgepole Pine, 5=Ponderosa Pine, 6=Spruce/Fir
COVER_TYPE_LABELS = {
    0: "Aspen",
    1: "Cottonwood / Willow",
    2: "Douglas-fir",
    3: "Krummholz",
    4: "Lodgepole Pine",
    5: "Ponderosa Pine",
    6: "Spruce / Fir",
}
COVER_TYPE_ICONS = {
    0: "🍂", 1: "🌿", 2: "🌲", 3: "🏔️", 4: "🌲", 5: "🌳", 6: "🌲",
}

SOIL_TYPES      = [str(i) for i in range(1, 41)]
WILDERNESS_AREAS = ["1", "2", "3", "4"]

SCALING_COLS = [
    "elevation", "aspect", "slope",
    "horizontal_dist_to_water", "vertical_dist_to_water",
    "horizontal_distance_to_roadways",
    "hillshade_9am", "hillshade_noon",
    "horizontal_dist_to_fire",
    "total_distance", "avg_hillshade",
    "water_distance", "shade_range", "elevation_slope",
]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🌲 Forest Cover Type Prediction ")
st.markdown("Enter the values for the following features to predict the forest cover type:")
st.divider()

col1, col2 = st.columns(2)

with col1:
    elevation = st.number_input("Elevation (meters)", min_value=0, max_value=4000, value=2599, step=1)
    horizontal_dist_to_water = st.number_input("Horizontal Distance To Hydrology", min_value=0, max_value=1500, value=300, step=10)
    horizontal_distance_to_roadways = st.number_input("Horizontal Distance To Roadways", min_value=0, max_value=7000, value=500, step=10)
    horizontal_dist_to_fire = st.number_input("Horizontal Distance To Fire Points", min_value=0, max_value=7000, value=600, step=10)
    vertical_dist_to_water = st.number_input("Vertical Distance To Hydrology", min_value=-500, max_value=700, value=50, step=1)

with col2:
    hillshade_9am  = st.slider("Hillshade 9am",  min_value=0, max_value=255, value=150)
    hillshade_noon = st.slider("Hillshade Noon", min_value=0, max_value=255, value=200)
    aspect = st.slider("Aspect (degrees)", min_value=0, max_value=360, value=90)
    slope  = st.slider("Slope (degrees)",  min_value=0, max_value=90,  value=15)

col3, col4 = st.columns(2)
with col3:
    wilderness_area = st.selectbox("Wilderness Area", options=WILDERNESS_AREAS, index=0,
                                   help="Which of the 4 wilderness areas (1–4)")
with col4:
    soil_type = st.selectbox("Soil Type", options=SOIL_TYPES, index=9,
                             help="Soil type category (1–40)")

# st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
def predict(inputs: dict):
    # Engineer features (hillshade_3pm was dropped in training)
    inputs["total_distance"] = (
        inputs["horizontal_dist_to_fire"]
        + inputs["horizontal_distance_to_roadways"]
        + inputs["horizontal_dist_to_water"]
    )
    inputs["avg_hillshade"]   = (inputs["hillshade_9am"] + inputs["hillshade_noon"]) / 2
    inputs["water_distance"]  = np.sqrt(
        inputs["horizontal_dist_to_water"]**2 + inputs["vertical_dist_to_water"]**2
    )
    inputs["shade_range"]     = abs(inputs["hillshade_9am"] - inputs["hillshade_noon"])
    inputs["elevation_slope"] = inputs["elevation"] * inputs["slope"]

    df = pd.DataFrame([inputs])
    df["wilderness_area"] = int(wilderness_area)
    df["soil_type"]       = int(soil_type)

    # OHE — same settings as training (drop_first=True)
    df_ohe = pd.get_dummies(df, columns=["soil_type", "wilderness_area"], drop_first=True)

    # Align to training column order
    df_ohe = df_ohe.reindex(columns=columns, fill_value=0)

    # Bool → int
    bool_cols = df_ohe.select_dtypes(include="bool").columns
    df_ohe[bool_cols] = df_ohe[bool_cols].astype(int)

    # Scale
    df_scaled = df_ohe.copy()
    scale_present = [c for c in SCALING_COLS if c in df_scaled.columns]
    df_scaled[scale_present] = scaler.transform(df_ohe[scale_present])

    pred  = model.predict(df_scaled)[0]
    label = COVER_TYPE_LABELS.get(int(pred), f"Type {pred}")
    icon  = COVER_TYPE_ICONS.get(int(pred), "🌲")
    return label, icon


if artifacts_loaded:
    if st.button("🔍 Predict Forest Cover Type", use_container_width=True, type="primary"):
        raw = {
            "elevation": elevation,
            "aspect": aspect,
            "slope": slope,
            "horizontal_dist_to_water": horizontal_dist_to_water,
            "vertical_dist_to_water": vertical_dist_to_water,
            "horizontal_distance_to_roadways": horizontal_distance_to_roadways,
            "hillshade_9am": hillshade_9am,
            "hillshade_noon": hillshade_noon,
            "horizontal_dist_to_fire": horizontal_dist_to_fire,
        }
        with st.spinner("Predicting..."):
            label, icon = predict(raw)

        st.success(f"{icon} **Predicted Forest Cover Type: {label}**")
else:
    st.info("Please ensure the model files are present to enable predictions.")
