
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import base64
from pathlib import Path

# ==================================================
# CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Forest Cover Type Prediction",
    page_icon="🌲",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "soil_cover_type.csv"
DEMO_FILE = BASE_DIR / "demo_examples.csv"

# ==================================================
# BACKGROUND
# ==================================================
def set_bg(image_file):
    if not image_file.exists():
        return

    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

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
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


set_bg(BASE_DIR / "forest_bg.png")

# ==================================================
# LOAD MODEL ARTIFACTS
# ==================================================
@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "random_forest.pkl", "rb") as f:
        model = pickle.load(f)

    scaler = joblib.load(BASE_DIR / "scaler.pkl")

    with open(BASE_DIR / "columns.pkl", "rb") as f:
        columns = pickle.load(f)

    with open(BASE_DIR / "power_transformer.pkl", "rb") as f:
        pt = pickle.load(f)

    with open(BASE_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    return model, scaler, list(columns), pt, le


try:
    model, scaler, columns, pt, le = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    st.error(f"Could not load model artifacts: {e}")


# ==================================================
# FEATURES: MUST MATCH TRAINING
# ==================================================
sqrt_cols = [
    "aspect",
    "slope",
    "horizontal_dist_to_water",
]

yeo_cols = [
    "vertical_dist_to_water",
    "hillshade_9am",
    "hillshade_noon",
    "elevation",
]

scaling_cols = [
    "elevation",
    "aspect",
    "slope",
    "horizontal_dist_to_water",
    "vertical_dist_to_water",
    "horizontal_distance_to_roadways",
    "hillshade_9am",
    "hillshade_noon",
    "horizontal_dist_to_fire",
    "total_distance",
    "avg_hillshade",
    "water_distance",
    "shade_range",
    "elevation_slope",
]

RAW_FEATURES = [
    "elevation",
    "aspect",
    "slope",
    "horizontal_dist_to_water",
    "vertical_dist_to_water",
    "horizontal_distance_to_roadways",
    "horizontal_dist_to_fire",
    "hillshade_9am",
    "hillshade_noon",
    "hillshade_3pm",
]

DEMO_INPUT_COLUMNS = RAW_FEATURES + [
    "wilderness_area",
    "soil_type",
]


# ==================================================
# CLEAN ORIGINAL DATASET COLUMN NAMES
# ==================================================
def clean_dataset(df):
    df = df.copy()

    df = df.rename(columns={
        "Horizontal_Distance_To_Hydrology":
            "Horizontal_Dist_To_water",
        "Vertical_Distance_To_Hydrology":
            "Vertical_Dist_To_water",
        "Horizontal_Distance_To_Fire_Points":
            "Horizontal_Dist_To_Fire",
    })

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    if "cover_type" in df.columns:
        df["cover_type"] = (
            df["cover_type"].astype(str).str.strip().str.title()
        )

    return df


# ==================================================
# PREPROCESS INPUTS
# Same transformations and feature engineering
# as the training notebook
# ==================================================
def prepare_model_input(raw_df):
    df = raw_df[DEMO_INPUT_COLUMNS].copy()

    # 1. Square-root transformations
    if (df[sqrt_cols] < 0).any().any():
        raise ValueError(
            "Aspect, slope and horizontal water distance "
            "must be non-negative."
        )

    df[sqrt_cols] = np.sqrt(df[sqrt_cols])

    # 2. Saved Yeo-Johnson transformation
    df[yeo_cols] = pt.transform(df[yeo_cols])

    # 3. Feature engineering
    df["total_distance"] = (
        df["horizontal_dist_to_fire"]
        + df["horizontal_distance_to_roadways"]
        + df["horizontal_dist_to_water"]
    )

    df["avg_hillshade"] = (
        df["hillshade_9am"]
        + df["hillshade_noon"]
        + df["hillshade_3pm"]
    ) / 3

    df["water_distance"] = np.sqrt(
        df["horizontal_dist_to_water"] ** 2
        + df["vertical_dist_to_water"] ** 2
    )

    shade_cols = [
        "hillshade_9am",
        "hillshade_noon",
        "hillshade_3pm",
    ]

    df["shade_range"] = (
        df[shade_cols].max(axis=1)
        - df[shade_cols].min(axis=1)
    )

    df["elevation_slope"] = df["elevation"] * df["slope"]

    # Drop 3 PM shade only after feature engineering
    df = df.drop(columns=["hillshade_3pm"])

    # 4. One-hot encode categories
    df["soil_type"] = df["soil_type"].astype(int)
    df["wilderness_area"] = df["wilderness_area"].astype(int)

    df_ohe = pd.get_dummies(
        df,
        columns=["soil_type", "wilderness_area"],
        drop_first=True,
    )

    # 5. Match exact training columns and order
    df_ohe = df_ohe.reindex(
        columns=columns,
        fill_value=0,
    )

    bool_cols = df_ohe.select_dtypes(include="bool").columns
    df_ohe[bool_cols] = df_ohe[bool_cols].astype(int)

    # 6. Apply saved scaler
    df_scaled = df_ohe.copy()

    df_scaled.loc[:, scaling_cols] = scaler.transform(
        df_ohe[scaling_cols]
    )

    return df_scaled


# ==================================================
# PREDICTION FUNCTION
# ==================================================
def predict(inputs, wilderness_area, soil_type):
    raw = inputs.copy()

    raw["wilderness_area"] = int(wilderness_area)
    raw["soil_type"] = int(soil_type)

    raw_df = pd.DataFrame([raw])
    model_input = prepare_model_input(raw_df)

    pred = model.predict(model_input)[0]

    predicted_label = le.inverse_transform(
        np.asarray([int(pred)])
    )[0]

    probabilities = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(model_input)[0]

        probabilities = pd.DataFrame({
            "Cover Type": le.inverse_transform(
                model.classes_.astype(int)
            ),
            "Probability (%)": np.round(probs * 100, 2),
        }).sort_values(
            "Probability (%)",
            ascending=False,
        ).reset_index(drop=True)

    return predicted_label, probabilities, model_input


# ==================================================
# CREATE VERIFIED DEMO CSV AUTOMATICALLY
# Uses real rows from soil_cover_type.csv
# ==================================================
def create_demo_examples():
    if DEMO_FILE.exists():
        return pd.read_csv(DEMO_FILE)

    if not DATA_FILE.exists():
        return pd.DataFrame()

    data = pd.read_csv(DATA_FILE)
    data = clean_dataset(data)

    required = DEMO_INPUT_COLUMNS + ["cover_type"]
    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(
            "Original dataset is missing required columns: "
            + ", ".join(missing)
        )

    data = data.dropna(subset=required).copy()

    # Categories and numeric inputs must be usable by the model
    for col in DEMO_INPUT_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=DEMO_INPUT_COLUMNS)

    examples = []

    # Find one correctly predicted real row per class
    for cover_type in sorted(data["cover_type"].unique()):
        group = data[data["cover_type"] == cover_type]

        # Try up to 2,000 real rows for each class
        candidates = group.sample(
            n=min(len(group), 2000),
            random_state=42,
        )

        candidate_inputs = candidates[DEMO_INPUT_COLUMNS].copy()

        # Use the exact same app preprocessing
        prepared = prepare_model_input(candidate_inputs)

        predictions = le.inverse_transform(
            model.predict(prepared).astype(int)
        )

        correct_positions = np.flatnonzero(
            predictions == cover_type
        )

        if len(correct_positions) > 0:
            chosen = candidates.iloc[
                int(correct_positions[0])
            ]

            examples.append({
                "cover_type": cover_type,
                **{
                    col: chosen[col]
                    for col in DEMO_INPUT_COLUMNS
                },
            })

    if not examples:
        return pd.DataFrame()

    result = pd.DataFrame(examples)
    result.to_csv(DEMO_FILE, index=False)

    return result


# ==================================================
# LOAD OR GENERATE DEMO EXAMPLES
# ==================================================
demo_df = pd.DataFrame()

if artifacts_loaded:
    try:
        demo_df = create_demo_examples()
    except Exception as e:
        st.warning(f"Demo examples could not be prepared: {e}")


# ==================================================
# USER INTERFACE
# ==================================================
st.title("🌲 Forest Cover Type Prediction")

st.write(
    "Predict a forest cover type using terrain, "
    "distance, hillshade, soil and wilderness features."
)

st.divider()

# --------------------------------------------------
# Demo selector
# --------------------------------------------------
st.subheader("🎯 Presentation Demo")

demo_options = {}

if not demo_df.empty and "cover_type" in demo_df.columns:
    demo_options = {
        str(row["cover_type"]): row
        for _, row in demo_df.iterrows()
    }

if demo_options:
    selected_demo = st.selectbox(
        "Choose a verified prediction example",
        ["Manual input"] + list(demo_options.keys()),
        key="selected_demo",
    )

    if st.button(
        "Load selected scenario",
        use_container_width=True,
    ):
        if selected_demo != "Manual input":
            example = demo_options[selected_demo]

            # Set widget values BEFORE those widgets are created
            for col in RAW_FEATURES:
                st.session_state[f"input_{col}"] = float(
                    example[col]
                )

            st.session_state["input_wilderness_area"] = str(
                int(example["wilderness_area"])
            )

            st.session_state["input_soil_type"] = str(
                int(example["soil_type"])
            )

            st.session_state["scenario_loaded"] = selected_demo

    if st.session_state.get("scenario_loaded"):
        st.info(
            "Loaded example: "
            + st.session_state["scenario_loaded"]
            + ". You can now change the inputs manually."
        )
else:
    st.info(
        "Demo scenarios are unavailable. Check that "
        "soil_cover_type.csv and all five model artifacts "
        "are in the project folder."
    )


# --------------------------------------------------
# Default input values
# --------------------------------------------------
defaults = {
    "input_elevation": 2599,
    "input_aspect": 90,
    "input_slope": 15,
    "input_horizontal_dist_to_water": 300,
    "input_vertical_dist_to_water": 50,
    "input_horizontal_distance_to_roadways": 500,
    "input_horizontal_dist_to_fire": 600,
    "input_hillshade_9am": 150,
    "input_hillshade_noon": 200,
    "input_hillshade_3pm": 150,
    "input_wilderness_area": "1",
    "input_soil_type": "10",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# Input widgets
# --------------------------------------------------
st.subheader("Environmental and Geographical Inputs")

col1, col2 = st.columns(2)

with col1:
    elevation = st.number_input(
        "Elevation (meters)",
        min_value=0,
        max_value=4000,
        key="input_elevation",
    )

    horizontal_dist_to_water = st.number_input(
        "Horizontal Distance To Hydrology",
        min_value=0,
        max_value=1500,
        key="input_horizontal_dist_to_water",
    )

    vertical_dist_to_water = st.number_input(
        "Vertical Distance To Hydrology",
        min_value=-500,
        max_value=700,
        key="input_vertical_dist_to_water",
    )

    horizontal_distance_to_roadways = st.number_input(
        "Horizontal Distance To Roadways",
        min_value=0,
        max_value=7000,
        key="input_horizontal_distance_to_roadways",
    )

    horizontal_dist_to_fire = st.number_input(
        "Horizontal Distance To Fire Points",
        min_value=0,
        max_value=7000,
        key="input_horizontal_dist_to_fire",
    )

with col2:
    aspect = st.slider(
        "Aspect (degrees)",
        0,
        360,
        key="input_aspect",
    )

    slope = st.slider(
        "Slope (degrees)",
        0,
        90,
        key="input_slope",
    )

    hillshade_9am = st.slider(
        "Hillshade 9 AM",
        0,
        255,
        key="input_hillshade_9am",
    )

    hillshade_noon = st.slider(
        "Hillshade Noon",
        0,
        255,
        key="input_hillshade_noon",
    )

    hillshade_3pm = st.slider(
        "Hillshade 3 PM",
        0,
        255,
        key="input_hillshade_3pm",
    )

col3, col4 = st.columns(2)

with col3:
    wilderness_area = st.selectbox(
        "Wilderness Area",
        ["1", "2", "3", "4"],
        key="input_wilderness_area",
    )

with col4:
    soil_type = st.selectbox(
        "Soil Type",
        [str(i) for i in range(1, 41)],
        key="input_soil_type",
    )


# --------------------------------------------------
# Predict
# --------------------------------------------------
if artifacts_loaded:
    if st.button(
        "🔍 Predict Forest Cover Type",
        type="primary",
        use_container_width=True,
    ):
        raw = {
            "elevation": elevation,
            "aspect": aspect,
            "slope": slope,
            "horizontal_dist_to_water":
                horizontal_dist_to_water,
            "vertical_dist_to_water":
                vertical_dist_to_water,
            "horizontal_distance_to_roadways":
                horizontal_distance_to_roadways,
            "horizontal_dist_to_fire":
                horizontal_dist_to_fire,
            "hillshade_9am": hillshade_9am,
            "hillshade_noon": hillshade_noon,
            "hillshade_3pm": hillshade_3pm,
        }

        try:
            with st.spinner("Predicting..."):
                label, probabilities, model_input = predict(
                    raw,
                    wilderness_area,
                    soil_type,
                )

            st.success(
                f"🌲 Predicted Forest Cover Type: **{label}**"
            )

            if probabilities is not None:
                st.subheader("Class Probabilities")

                st.dataframe(
                    probabilities,
                    use_container_width=True,
                    hide_index=True,
                )

                st.bar_chart(
                    probabilities.set_index("Cover Type")
                )

            with st.expander("View processed model input"):
                st.dataframe(model_input)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
