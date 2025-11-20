# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import io
import os
from datetime import datetime
import h5py
from pathlib import Path

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Liver Cancer Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Styles ----------------
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

# ---------------- Constants ----------------
LOCAL_MODEL_FILES = {
    "CNN": "best_cnn_simple.h5",
    "MobileNetV2": "best_mobilenetv2.h5",
    "EfficientNetB0": "best_efficientnetb0.h5"
}
SCALER_PATH = "scaler_clinical.pkl"
HISTORY_FILE = "predictions_history.csv"

# ---------------- Utilities: history ----------------
def load_history() -> pd.DataFrame:
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

def append_history(record: dict):
    df_hist = load_history()
    row = pd.DataFrame([record])
    df_hist = pd.concat([df_hist, row], ignore_index=True)
    try:
        df_hist.to_csv(HISTORY_FILE, index=False)
    except Exception as e:
        st.error(f"Could not save history: {e}")

def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

# ---------------- Utilities: file checks & loading ----------------
def is_h5_valid(path: str):
    try:
        with h5py.File(path, 'r'):
            return True, None
    except Exception as e:
        return False, str(e)

@st.cache_resource
def load_scaler(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        return None

@st.cache_resource
def try_load_model(path: str):
    try:
        model = tf.keras.models.load_model(path)
        return model, None
    except Exception as e:
        return None, str(e)

# ---------------- Sidebar: files & defaults ----------------
st.sidebar.header("Files & Model")

# show status for each local model
models_loaded = {}
uploaded_models_tmp = {}

for key, local_path in LOCAL_MODEL_FILES.items():
    if Path(local_path).exists():
        valid, err = is_h5_valid(local_path)
        if not valid:
            st.sidebar.warning(f"{key}: local file found but invalid: {err}")
        else:
            model, load_err = try_load_model(local_path)
            if model is None:
                st.sidebar.error(f"{key}: failed to load local model: {load_err}")
            else:
                st.sidebar.success(f"{key}: local model loaded")
                models_loaded[key] = model
    else:
        st.sidebar.info(f"{key}: local file not found")

st.sidebar.markdown("---")
st.sidebar.info("If a local model file is missing/corrupt you can upload a .h5 file to use temporarily.")

# uploader for any .h5 to use temporarily (it will override the corresponding model in-memory)
uploaded_model_file = st.sidebar.file_uploader("Upload a model (.h5) to use temporarily", type=["h5"])
if uploaded_model_file is not None:
    tmp_path = "temp_uploaded_model.h5"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_model_file.getbuffer())
    model_tmp, load_err = try_load_model(tmp_path)
    if model_tmp is None:
        st.sidebar.error(f"Uploaded model couldn't be loaded: {load_err}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    else:
        st.sidebar.success("Uploaded model loaded (will be used for prediction).")
        # put it under a generic key so user can pick it later
        models_loaded["UploadedModel"] = model_tmp
        uploaded_models_tmp["UploadedModel"] = tmp_path

# load scaler
scaler = load_scaler(SCALER_PATH)
if scaler is None:
    st.sidebar.error("Scaler not loaded (scaler_clinical.pkl missing or invalid).")
else:
    st.sidebar.success("Scaler loaded.")

st.sidebar.markdown("---")
st.sidebar.header("Default clinical values (editable)")

# initialize defaults in session_state
default_values = {
    "age": 50,
    "sex": "Male",
    "hepatitis": "No virus",
    "smoking": "No",
    "alcohol": "No",
    "diabetes": "No",
    "fhx_can": "No",
    "fhx_livc": "No",
    "evidence_of_cirh": "No",
    "cps": "A",
    "afp": 10.0,
    "tr_size": 2.0,
    "tumor_nodul": "Uninodular"
}
for k, v in default_values.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.sidebar.checkbox("Edit default values"):
    st.session_state.age = st.sidebar.number_input("Default Age", min_value=0, max_value=120, value=int(st.session_state.age))
    st.session_state.sex = st.sidebar.selectbox("Default Sex", ["Male", "Female"], index=0 if st.session_state.sex=="Male" else 1)
    st.session_state.hepatitis = st.sidebar.selectbox(
        "Default Hepatitis", ["No virus","HBV only","HCV only","HCV and HBV"],
        index=["No virus","HBV only","HCV only","HCV and HBV"].index(st.session_state.hepatitis)
    )
    st.session_state.smoking = st.sidebar.selectbox("Default Smoking", ["No","Yes"], index=0 if st.session_state.smoking=="No" else 1)
    st.session_state.alcohol = st.sidebar.selectbox("Default Alcohol", ["No","Yes"], index=0 if st.session_state.alcohol=="No" else 1)
    st.session_state.diabetes = st.sidebar.selectbox("Default Diabetes", ["No","Yes"], index=0 if st.session_state.diabetes=="No" else 1)
    st.session_state.fhx_can = st.sidebar.selectbox("Default Family history cancer", ["No","Yes"], index=0 if st.session_state.fhx_can=="No" else 1)
    st.session_state.fhx_livc = st.sidebar.selectbox("Default Family history liver cancer", ["No","Yes"], index=0 if st.session_state.fhx_livc=="No" else 1)
    st.session_state.evidence_of_cirh = st.sidebar.selectbox("Default Evidence cirrhosis", ["No","Yes"], index=0 if st.session_state.evidence_of_cirh=="No" else 1)
    st.session_state.cps = st.sidebar.selectbox("Default CPS", ["A","B","C"], index=["A","B","C"].index(st.session_state.cps))
    st.session_state.afp = st.sidebar.number_input("Default AFP (ng/ml)", value=float(st.session_state.afp), format="%.2f")
    st.session_state.tr_size = st.sidebar.number_input("Default Tumor size (cm)", value=float(st.session_state.tr_size), format="%.2f")
    st.session_state.tumor_nodul = st.sidebar.selectbox("Default Tumor nodularity", ["Uninodular","Multinodular"],
                                                       index=0 if st.session_state.tumor_nodul=="Uninodular" else 1)

st.sidebar.markdown("---")
st.sidebar.header("Prediction History")
history_df = load_history()
if history_df.empty:
    st.sidebar.write("No saved predictions yet.")
else:
    st.sidebar.write(f"Saved predictions: {len(history_df)}")
    st.sidebar.dataframe(history_df.sort_values(by="timestamp", ascending=False).head(8))
    st.sidebar.download_button(
        "Download full history (CSV)",
        history_df.to_csv(index=False).encode('utf-8'),
        file_name="predictions_history.csv",
        mime="text/csv"
    )
    if st.sidebar.button("Clear history"):
        clear_history()
        st.sidebar.success("History cleared.")
        history_df = load_history()

# ---------------- Header ----------------
col_title, col_logo = st.columns([8,1])
with col_title:
    st.markdown('<div class="big-title">Liver Cancer Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a CT image and enter patient clinical data to get a prediction.</div>', unsafe_allow_html=True)

# ---------------- Main layout ----------------
left_col, right_col = st.columns([1,1])

# Left: image upload & preview
with left_col:
    st.subheader("Image")
    uploaded_file = st.file_uploader("Upload CT image (PNG / JPG)", type=["png", "jpg", "jpeg"])
    image = None
    if uploaded_file:
        try:
            image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
            display_image = image.copy()
            display_image.thumbnail((400,400))
            st.image(display_image, caption="Uploaded CT image", use_column_width=True)
        except Exception as e:
            st.error(f"Error reading image: {e}")
            uploaded_file = None
    else:
        st.info("No image uploaded yet. Please upload a CT image (PNG/JPG).")

    st.markdown("---")
    st.write("Model selection")
    model_options = list(models_loaded.keys())
    if not model_options:
        st.warning("No models available. Upload a .h5 on the sidebar or place model files in the app folder.")
    model_name = st.selectbox("Choose model", model_options if model_options else ["None"])
    predict_button = st.button("Predict")

# Right: clinical inputs (no preview)
with right_col:
    st.subheader("Clinical information")
    use_same = st.checkbox("Use same patient values (use last entered / defaults)")

    # prefill from session_state
    if use_same:
        age = st.number_input("Age", min_value=0, max_value=120, value=int(st.session_state.age))
        sex = st.selectbox("Sex", ["Male", "Female"], index=0 if st.session_state.sex=="Male" else 1)
        hepatitis = st.selectbox("Type of hepatitis", ["No virus", "HBV only", "HCV only", "HCV and HBV"],
                                 index=["No virus", "HBV only", "HCV only", "HCV and HBV"].index(st.session_state.hepatitis))
        smoking = st.selectbox("Smoking", ["No", "Yes"], index=0 if st.session_state.smoking=="No" else 1)
        alcohol = st.selectbox("Alcohol consumption", ["No", "Yes"], index=0 if st.session_state.alcohol=="No" else 1)
        diabetes = st.selectbox("Diabetes", ["No", "Yes"], index=0 if st.session_state.diabetes=="No" else 1)
        fhx_can = st.selectbox("Family history of cancer", ["No", "Yes"], index=0 if st.session_state.fhx_can=="No" else 1)
        fhx_livc = st.selectbox("Family history of liver cancer", ["No", "Yes"], index=0 if st.session_state.fhx_livc=="No" else 1)
        evidence_of_cirh = st.selectbox("Evidence of cirrhosis", ["No", "Yes"], index=0 if st.session_state.evidence_of_cirh=="No" else 1)
        cps = st.selectbox("Child-Pugh Score (CPS)", ["A", "B", "C"], index=["A","B","C"].index(st.session_state.cps))
        afp = st.number_input("AFP (alpha-fetoprotein) (ng/ml)", min_value=0.0, value=float(st.session_state.afp), format="%.2f")
        tr_size = st.number_input("Tumor size (cm)", min_value=0.0, value=float(st.session_state.tr_size), format="%.2f")
        tumor_nodul = st.selectbox("Tumor nodularity", ["Uninodular", "Multinodular"], index=0 if st.session_state.tumor_nodul=="Uninodular" else 1)
    else:
        age = st.number_input("Age", min_value=0, max_value=120, value=int(st.session_state.age))
        sex = st.selectbox("Sex", ["Male", "Female"], index=0 if st.session_state.sex=="Male" else 1)
        hepatitis = st.selectbox("Type of hepatitis", ["No virus", "HBV only", "HCV only", "HCV and HBV"],
                                 index=["No virus", "HBV only", "HCV only", "HCV and HBV"].index(st.session_state.hepatitis))
        smoking = st.selectbox("Smoking", ["No", "Yes"], index=0 if st.session_state.smoking=="No" else 1)
        alcohol = st.selectbox("Alcohol consumption", ["No", "Yes"], index=0 if st.session_state.alcohol=="No" else 1)
        diabetes = st.selectbox("Diabetes", ["No", "Yes"], index=0 if st.session_state.diabetes=="No" else 1)
        fhx_can = st.selectbox("Family history of cancer", ["No", "Yes"], index=0 if st.session_state.fhx_can=="No" else 1)
        fhx_livc = st.selectbox("Family history of liver cancer", ["No", "Yes"], index=0 if st.session_state.fhx_livc=="No" else 1)
        evidence_of_cirh = st.selectbox("Evidence of cirrhosis", ["No", "Yes"], index=0 if st.session_state.evidence_of_cirh=="No" else 1)
        cps = st.selectbox("Child-Pugh Score (CPS)", ["A", "B", "C"], index=["A","B","C"].index(st.session_state.cps))
        afp = st.number_input("AFP (alpha-fetoprotein) (ng/ml)", min_value=0.0, value=float(st.session_state.afp), format="%.2f")
        tr_size = st.number_input("Tumor size (cm)", min_value=0.0, value=float(st.session_state.tr_size), format="%.2f")
        tumor_nodul = st.selectbox("Tumor nodularity", ["Uninodular", "Multinodular"], index=0 if st.session_state.tumor_nodul=="Uninodular" else 1)

    # Save last patient values for reuse
    if st.button("Save this patient as 'last used'"):
        st.session_state.age = age
        st.session_state.sex = sex
        st.session_state.hepatitis = hepatitis
        st.session_state.smoking = smoking
        st.session_state.alcohol = alcohol
        st.session_state.diabetes = diabetes
        st.session_state.fhx_can = fhx_can
        st.session_state.fhx_livc = fhx_livc
        st.session_state.evidence_of_cirh = evidence_of_cirh
        st.session_state.cps = cps
        st.session_state.afp = afp
        st.session_state.tr_size = tr_size
        st.session_state.tumor_nodul = tumor_nodul
        st.success("Saved last patient values to session.")

# ---------------- Prepare clinical dataframe & scaling ----------------
clin_dict = {
    "age": age,
    "Sex": 1 if sex == "Male" else 2,
    "hepatitis": {"No virus":0, "HBV only":1, "HCV only":2, "HCV and HBV":3}[hepatitis],
    "Smoking": 1 if smoking == "Yes" else 0,
    "Alcohol": 1 if alcohol == "Yes" else 0,
    "fhx_can": 1 if fhx_can == "Yes" else 0,
    "fhx_livc": 1 if fhx_livc == "Yes" else 0,
    "Diabetes": 1 if diabetes == "Yes" else 0,
    "Evidence_of_cirh": 1 if evidence_of_cirh == "Yes" else 0,
    "CPS": {"A":1, "B":2, "C":3}[cps],
    "AFP": float(afp),
    "Tr_Size": float(tr_size),
    "tumor_nodul": 0 if tumor_nodul == "Uninodular" else 1
}
clin_df = pd.DataFrame([clin_dict])

clin_scaled = None
if scaler is not None:
    try:
        # add missing features with zeros
        for col in scaler.feature_names_in_:
            if col not in clin_df.columns:
                clin_df[col] = 0
        clin_df = clin_df[scaler.feature_names_in_]
        clin_scaled = scaler.transform(clin_df).astype("float32")
    except Exception as e:
        st.error(f"Error preparing clinical features for scaler: {e}")
        clin_scaled = None

# ---------------- Functions: prepare image ----------------
def prepare_image_for_model(pil_img: Image.Image, target_size=(128,128)):
    img = pil_img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch dim
    return arr

# ---------------- Prediction action ----------------
if predict_button:
    if model_name not in models_loaded:
        st.error("Selected model is not available. Upload a model or place model files in the app folder.")
    elif uploaded_file is None or image is None:
        st.warning("Please upload an image before predicting.")
    elif scaler is None or clin_scaled is None:
        st.error("Scaler is not available or clinical data couldn't be processed.")
    else:
        model = models_loaded[model_name]
        try:
            img_array = prepare_image_for_model(image, target_size=(128,128))
            # Try common predict patterns
            raw_pred = None
            try:
                raw_pred = model.predict([img_array, clin_scaled], verbose=0)
            except Exception:
                try:
                    raw_pred = model.predict([clin_scaled, img_array], verbose=0)
                except Exception:
                    raw_pred = model.predict(img_array, verbose=0)

            if isinstance(raw_pred, (list, tuple)):
                raw_pred = raw_pred[0]
            prob = float(np.squeeze(raw_pred))
            label = "Progressed (Positive)" if prob > 0.5 else "No Progression (Negative)"
            pct = prob * 100

            # Show result
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.subheader("Prediction result")
            st.metric("Prediction", label, delta=f"{pct:.2f}% probability")
            st.write(f"Raw probability score: **{prob:.4f}**")
            if prob > 0.5:
                st.success("Model indicates a positive/progression prediction. Consult clinical specialists for confirmation.")
            else:
                st.info("Model indicates a negative/no-progression prediction. Clinical correlation recommended.")
            st.markdown("</div>", unsafe_allow_html=True)

            # Save to history
            record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": model_name,
                "label": label,
                "probability": float(prob),
                "age": age,
                "Sex": sex,
                "hepatitis": hepatitis,
                "Smoking": smoking,
                "Alcohol": alcohol,
                "Diabetes": diabetes,
                "AFP": float(afp),
                "Tr_Size": float(tr_size),
                "tumor_nodul": tumor_nodul
            }
            try:
                append_history(record)
                st.info("Prediction saved to history.")
            except Exception as e:
                st.error(f"Could not save prediction: {e}")

            # refresh in-memory history
            history_df = load_history()

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------- Bottom: full history view & controls ----------------
st.markdown("---")
st.header("Full prediction history")
hist = load_history()
if hist.empty:
    st.write("No saved predictions yet.")
else:
    st.dataframe(hist.sort_values(by="timestamp", ascending=False))
    st.download_button("Download history (CSV)", hist.to_csv(index=False).encode('utf-8'), file_name="predictions_history.csv")
    if st.button("Clear history (main)"):
        clear_history()
        st.success("History cleared.")

st.caption("Note: This tool is a demonstration. Do not use this app as a substitute for professional medical diagnosis. If you deploy this app on ephemeral hosting (Streamlit Cloud, Heroku), the CSV file may not persist — consider using a database for durable storage.")

