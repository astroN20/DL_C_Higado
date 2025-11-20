import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="Liver Cancer Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styles (simple) ---
st.markdown(
    """
    <style>
    .stApp { font-family: "Inter", sans-serif; }
    .big-title { font-size:28px; font-weight:700; margin-bottom: 0.25rem; }
    .subtitle { color: #6b7280; margin-top: -8px; margin-bottom: 12px; }
    .result-box { border-radius:12px; padding:16px; box-shadow: 0 4px 30px rgba(0,0,0,0.05); }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Caching loaders for heavy resources ---
@st.cache_resource
def load_models():
    # Wrap loading in try/except so the app doesn't crash without files
    loaded = {}
    try:
        loaded["CNN"] = tf.keras.models.load_model("best_cnn_simple.h5")
    except Exception as e:
        loaded["CNN"] = None
        st.sidebar.error(f"Could not load CNN model: {e}")
    try:
        loaded["MobileNetV2"] = tf.keras.models.load_model("best_mobilenetv2.h5")
    except Exception as e:
        loaded["MobileNetV2"] = None
        st.sidebar.error(f"Could not load MobileNetV2 model: {e}")
    try:
        loaded["EfficientNetB0"] = tf.keras.models.load_model("best_efficientnetb0.h5")
    except Exception as e:
        loaded["EfficientNetB0"] = None
        st.sidebar.error(f"Could not load EfficientNetB0: {e}")
    return loaded

@st.cache_resource
def load_scaler():
    try:
        return joblib.load("scaler_clinical.pkl")
    except Exception as e:
        st.sidebar.error(f"Could not load scaler: {e}")
        return None

models = load_models()
scaler = load_scaler()

# --- Header ---
col_title, col_logo = st.columns([8,1])
with col_title:
    st.markdown('<div class="big-title">Liver Cancer Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a CT image and enter the patient clinical data to get a prediction.</div>', unsafe_allow_html=True)

# --- Sidebar: Inputs ---
st.sidebar.header("Input & Model Selection")

uploaded_file = st.sidebar.file_uploader("Upload CT image (PNG / JPG)", type=["png", "jpg", "jpeg"])
model_name = st.sidebar.selectbox("Choose model", ["CNN", "MobileNetV2", "EfficientNetB0"])
predict_button = st.sidebar.button("Predict")

# --- Main layout: image on left, clinical data on right ---
left_col, right_col = st.columns([1,1])

# Left: show uploaded image and preview
with left_col:
    st.subheader("Image")
    if uploaded_file:
        try:
            image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
            # Keep size consistent with your model training (128x128 here)
            display_image = image.copy()
            display_image.thumbnail((400,400))
            st.image(display_image, caption="Uploaded CT image", use_column_width=True)
        except Exception as e:
            st.error(f"Error reading image: {e}")
            uploaded_file = None
    else:
        st.info("No image uploaded yet. Please upload a chest/abdominal CT scan image (PNG/JPG).")

# Right: clinical inputs grouped in an expander
with right_col:
    st.subheader("Clinical information")
    with st.expander("Enter patient clinical variables", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=50)
            sex = st.selectbox("Sex", ["Male", "Female"])
            sex_code = 1 if sex == "Male" else 2
            hepatitis = st.selectbox("Type of hepatitis", ["No virus", "HBV only", "HCV only", "HCV and HBV"])
            hepatitis_code = {"No virus": 0, "HBV only": 1, "HCV only": 2, "HCV and HBV": 3}[hepatitis]
            smoking = st.selectbox("Smoking", ["No", "Yes"])
            smoking_code = 1 if smoking == "Yes" else 0
        with col2:
            alcohol = st.selectbox("Alcohol consumption", ["No", "Yes"])
            alcohol_code = 1 if alcohol == "Yes" else 0
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            diabetes_code = 1 if diabetes == "Yes" else 0
            fhx_can = st.selectbox("Family history of cancer", ["No", "Yes"])
            fhx_can_code = 1 if fhx_can == "Yes" else 0
            fhx_livc = st.selectbox("Family history of liver cancer", ["No", "Yes"])
            fhx_livc_code = 1 if fhx_livc == "Yes" else 0

        # Next row
        col3, col4 = st.columns(2)
        with col3:
            evidence_of_cirh = st.selectbox("Evidence of cirrhosis", ["No", "Yes"])
            evidence_of_cirh_code = 1 if evidence_of_cirh == "Yes" else 0
            cps = st.selectbox("Child-Pugh Score (CPS)", ["A", "B", "C"])
            cps_code = {"A": 1, "B": 2, "C": 3}[cps]
        with col4:
            afp = st.number_input("AFP (alpha-fetoprotein) (ng/ml)", min_value=0.0, value=10.0, format="%.2f")
            tr_size = st.number_input("Tumor size (cm)", min_value=0.0, value=2.0, format="%.2f")
            tumor_nodul = st.selectbox("Tumor nodularity", ["Uninodular", "Multinodular"])
            tumor_nodul_code = 0 if tumor_nodul == "Uninodular" else 1

# --- Build clinical DataFrame ---
clin_dict = {
    "age": age,
    "Sex": sex_code,
    "hepatitis": hepatitis_code,
    "Smoking": smoking_code,
    "Alcohol": alcohol_code,
    "fhx_can": fhx_can_code,
    "fhx_livc": fhx_livc_code,
    "Diabetes": diabetes_code,
    "Evidence_of_cirh": evidence_of_cirh_code,
    "CPS": cps_code,
    "AFP": afp,
    "Tr_Size": tr_size,
    "tumor_nodul": tumor_nodul_code
}
clin_df = pd.DataFrame([clin_dict])

# If scaler is available, align features and scale
clin_scaled = None
if scaler is not None:
    # add missing features with zeros
    try:
        for col in scaler.feature_names_in_:
            if col not in clin_df.columns:
                clin_df[col] = 0
        clin_df = clin_df[scaler.feature_names_in_]
        clin_scaled = scaler.transform(clin_df).astype("float32")
    except Exception as e:
        st.sidebar.error(f"Error preparing clinical features for scaler: {e}")
        clin_scaled = None

# --- Prediction logic ---
def prepare_image_for_model(pil_img, target_size=(128,128)):
    img = pil_img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch dim
    return arr

# Show clinical table (collapsible)
with st.expander("Preview clinical data (table)"):
    st.dataframe(clin_df.T.rename(columns={0: "value"}))

# When user clicks Predict:
if predict_button:
    # Basic checks
    if models.get(model_name) is None:
        st.error(f"The chosen model '{model_name}' is not loaded.")
    elif uploaded_file is None:
        st.warning("Please upload an image before predicting.")
    elif scaler is None or clin_scaled is None:
        st.error("Scaler is not available or clinical data couldn't be processed.")
    else:
        try:
            image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
            img_array = prepare_image_for_model(image, target_size=(128,128))
            model = models[model_name]

            # Some models expect two inputs (image + clinical). Adjust if necessary for your model signature.
            # We'll try to call predict([img_array, clin_scaled]) and fallback to img-only if it fails.
            try:
                raw_pred = model.predict([img_array, clin_scaled], verbose=0)
            except Exception:
                # try alternative ordering or single input
                try:
                    raw_pred = model.predict([clin_scaled, img_array], verbose=0)
                except Exception:
                    raw_pred = model.predict(img_array, verbose=0)

            # raw_pred may be shape (1,1) or (1,) etc.
            if isinstance(raw_pred, (list, tuple)):
                # if model returns multiple outputs, take first
                raw_pred = raw_pred[0]
            pred_value = float(np.squeeze(raw_pred))
            prob = pred_value
            label = "Progressed (Positive)" if prob > 0.5 else "No Progression (Negative)"
            pct = prob * 100

            # Result box
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.subheader("Prediction result")
            st.metric("Prediction", label, delta=f"{pct:.2f}% probability")
            st.write(f"Raw probability score: **{prob:.4f}**")
            if prob > 0.5:
                st.success("Model indicates a positive/progression prediction. Consult clinical specialists for confirmation.")
            else:
                st.info("Model indicates a negative/no-progression prediction. Clinical correlation recommended.")
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Footer / notes
st.markdown("---")
st.caption("Note: This tool is a demonstration. Do not use this app as a substitute for professional medical diagnosis. Models and scaler files must be available in the app folder.")
