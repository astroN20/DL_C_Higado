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
from pathlib import Path

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Liver Cancer Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- Utilities: history ----------------
HISTORY_FILE = "predictions_history.csv"

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

# ---------------- Safe loaders ----------------
def safe_load_model(path: str):
    try:
        return tf.keras.models.load_model(path)
    except Exception:
        return None

def safe_load_scaler(path: str):
    try:
        return joblib.load(path)
    except Exception:
        return None

# ---------------- Load models & scaler (silent) ----------------
MODEL_FILES = {
    "CNN": "best_cnn_simple.h5",
    "MobileNetV2": "best_mobilenetv2.h5",
    "EfficientNetB0": "best_efficientnetb0.h5"
}

models = {}
for name, p in MODEL_FILES.items():
    if Path(p).exists():
        m = safe_load_model(p)
        if m is not None:
            models[name] = m

# If no models could be loaded, stop quietly with minimal message
if not models:
    st.error("No models available. Place model files in the app folder.")
    st.stop()

scaler = safe_load_scaler("scaler_clinical.pkl")
if scaler is None:
    st.error("Scaler file missing or invalid (scaler_clinical.pkl).")
    st.stop()

# ---------------- App header ----------------
st.title("Liver Cancer Detection System")
st.markdown("Upload a CT image and enter clinical patient data to get a prediction. This is a demonstration tool — not a medical diagnosis system.")

# ---------------- Left: image upload & model select ----------------
left_col, right_col = st.columns([1,1])

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

    st.markdown("---")
    st.subheader("Model")
    model_name = st.selectbox("Choose model", list(models.keys()))
    predict_button = st.button("Predict")

# ---------------- Right: clinical inputs arranged side-by-side (two columns per row) ----------------
with right_col:
    st.subheader("Clinical information (two-column layout)")

    # initialize last-used values if not present
    defaults = {
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
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Row 1: Age | Sex
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=int(st.session_state.age))
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"], index=0 if st.session_state.sex=="Male" else 1)

    # Row 2: Hepatitis | Smoking
    col1, col2 = st.columns(2)
    with col1:
        hepatitis = st.selectbox(
            "Type of hepatitis",
            ["No virus", "HBV only", "HCV only", "HCV and HBV"],
            index=["No virus", "HBV only", "HCV only", "HCV and HBV"].index(st.session_state.hepatitis)
        )
    with col2:
        smoking = st.selectbox("Smoking", ["No", "Yes"], index=0 if st.session_state.smoking=="No" else 1)

    # Row 3: Alcohol | Diabetes
    col1, col2 = st.columns(2)
    with col1:
        alcohol = st.selectbox("Alcohol consumption", ["No", "Yes"], index=0 if st.session_state.alcohol=="No" else 1)
    with col2:
        diabetes = st.selectbox("Diabetes", ["No", "Yes"], index=0 if st.session_state.diabetes=="No" else 1)

    # Row 4: Family history (any) | Family history (liver)
    col1, col2 = st.columns(2)
    with col1:
        fhx_can = st.selectbox("Family history of cancer", ["No", "Yes"], index=0 if st.session_state.fhx_can=="No" else 1)
    with col2:
        fhx_livc = st.selectbox("Family history of liver cancer", ["No", "Yes"], index=0 if st.session_state.fhx_livc=="No" else 1)

    # Row 5: Evidence of cirrhosis | Child-Pugh Score
    col1, col2 = st.columns(2)
    with col1:
        evidence_of_cirh = st.selectbox("Evidence of cirrhosis", ["No", "Yes"], index=0 if st.session_state.evidence_of_cirh=="No" else 1)
    with col2:
        cps = st.selectbox("Child-Pugh Score (CPS)", ["A", "B", "C"], index=["A","B","C"].index(st.session_state.cps))

    # Row 6: AFP | Tumor size
    col1, col2 = st.columns(2)
    with col1:
        afp = st.number_input("AFP (alpha-fetoprotein) (ng/ml)", min_value=0.0, value=float(st.session_state.afp), format="%.2f")
    with col2:
        tr_size = st.number_input("Tumor size (cm)", min_value=0.0, value=float(st.session_state.tr_size), format="%.2f")

    # Row 7: Tumor nodularity (full width)
    tumor_nodul = st.selectbox("Tumor nodularity", ["Uninodular", "Multinodular"], index=0 if st.session_state.tumor_nodul=="Uninodular" else 1)

    

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
try:
    for col in scaler.feature_names_in_:
        if col not in clin_df.columns:
            clin_df[col] = 0
    clin_df = clin_df[scaler.feature_names_in_]
    clin_scaled = scaler.transform(clin_df).astype("float32")
except Exception as e:
    st.error("Error preparing clinical features for scaler.")
    st.stop()

# ---------------- helper: prepare image ----------------
def prepare_image_for_model(pil_img: Image.Image, target_size=(128,128)):
    img = pil_img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------- Prediction ----------------
# ---------------- Prediction (robust input handling, auto-save history) ----------------
if predict_button:
    if uploaded_file is None or image is None:
        st.warning("Please upload an image before predicting.")
    else:
        model = models.get(model_name)
        if model is None:
            st.error("Selected model is not available.")
        else:
            try:
                # prepare image at correct size expected by model
                img_array = prepare_image_for_model(image, target_size=(224,224))  # <-- 224x224

                if clin_scaled is None:
                    st.error("Clinical data not available for prediction.")
                    raise RuntimeError("No clinical data")

                # make sure clin_scaled shape/dtype are correct
                import numpy as np
                clin_scaled = np.asarray(clin_scaled).astype("float32")
                if clin_scaled.ndim == 1:
                    clin_scaled = np.expand_dims(clin_scaled, axis=0)

                # detect expected inputs of the model
                try:
                    n_inputs = len(model.inputs)
                except Exception:
                    n_inputs = 1

                # assemble inputs according to input shapes (image -> rank 4)
                pred_inputs = None
                if n_inputs == 1:
                    inp0 = model.inputs[0]
                    if len(inp0.shape) == 4:
                        pred_inputs = img_array
                    else:
                        pred_inputs = clin_scaled
                elif n_inputs == 2:
                    inp_list = [None, None]
                    for i, inp in enumerate(model.inputs):
                        try:
                            if len(inp.shape) == 4:
                                inp_list[i] = img_array
                            else:
                                inp_list[i] = clin_scaled
                        except Exception:
                            inp_list[i] = None
                    # fill missing slots sensibly
                    if inp_list[0] is None and inp_list[1] is None:
                        pred_inputs = [img_array, clin_scaled]
                    else:
                        for i in range(2):
                            if inp_list[i] is None:
                                inp_list[i] = img_array if i == 0 else clin_scaled
                        pred_inputs = inp_list
                else:
                    # fallback: pass image then clin
                    pred_inputs = [img_array, clin_scaled] + [np.zeros((1,))] * (n_inputs - 2)

                # call predict with the prepared inputs
                raw_pred = model.predict(pred_inputs, verbose=0)

                if isinstance(raw_pred, (list, tuple)):
                    raw_pred = raw_pred[0]
                prob = float(np.squeeze(raw_pred))
                label = "Progressed (Positive)" if prob > 0.5 else "No Progression (Negative)"
                pct = prob * 100

                st.subheader("Prediction result")
                st.metric("Prediction", label, delta=f"{pct:.2f}% probability")
                st.write(f"Raw probability score: **{prob:.4f}**")
                if prob > 0.5:
                    st.success("Model indicates a positive/progression prediction. Consult clinical specialists for confirmation.")
                else:
                    st.info("Model indicates a negative/no-progression prediction. Clinical correlation recommended.")

                # --- auto-save to history (only when prediction was successful) ---
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
                append_history(record)
                st.info("Prediction saved to history.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------------- Single history view (only here) ----------------
st.markdown("---")
st.header("Prediction History")
history_df = load_history()
if history_df.empty:
    st.write("No saved predictions yet.")
else:
    st.dataframe(history_df.sort_values(by="timestamp", ascending=False))
    st.download_button("Download history (CSV)", history_df.to_csv(index=False).encode('utf-8'), file_name="predictions_history.csv")
    if st.button("Clear history"):
        clear_history()
        st.success("History cleared.")
        # reload view
        history_df = load_history()

st.caption("Note: This is a demonstration tool. The CSV file is stored in the app folder; on ephemeral hosting it may not persist across restarts.")
