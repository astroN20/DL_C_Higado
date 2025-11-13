import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from PIL import Image

# --- Cargar modelos ---
models = {
    "CNN": tf.keras.models.load_model("best_cnn_simple.h5"),
    "MobileNetV2": tf.keras.models.load_model("best_mobilenetv2.h5"),
    "EfficientNetB0": tf.keras.models.load_model("best_efficientnetb0.h5")
}

# --- Cargar scaler ---
scaler = joblib.load("scaler_clinical.pkl")

st.title("🩺 Detección de Cáncer de Hígado — Deep Learning")
st.markdown("Carga una **imagen de TC** y los **datos clínicos del paciente** para predecir si el caso es *Parient Cesored* o *Progressed*.")

# --- Entrada de imagen ---
img_file = st.file_uploader("📷 Subir imagen (PNG o JPG)", type=["png", "jpg", "jpeg"])
if img_file:
    img = Image.open(img_file).convert("RGB").resize((128, 128))
    st.image(img, caption="Imagen cargada", use_column_width=True)
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

# --- Entradas clínicas ---
st.subheader("Datos clínicos")
clinical_inputs = {}
fields = [
    "age", "Sex", "hepatitis", "Smoking", "Alcohol", "fhx_can", "fhx_livc",
    "Diabetes", "Evidence_of_cirh", "3PS", "AFP", "Tr_Size", "tumor_nodul"
]
for f in fields:
    clinical_inputs[f] = st.number_input(f"Ingrese {f}", value=0.0)

clin_df = pd.DataFrame([clinical_inputs])
clin_scaled = scaler.transform(clin_df).astype("float32")

# --- Selección del modelo ---
model_name = st.selectbox("Selecciona el modelo a usar", list(models.keys()))

if st.button("🔍 Predecir"):
    model = models[model_name]
    if img_file:
        pred = model.predict([img_array, clin_scaled])[0][0]
        label = "Progressed (Positivo)" if pred > 0.5 else "Parient Cesored (Negativo)"
        st.subheader(f"Resultado: {label}")
        st.write(f"Probabilidad: {pred:.3f}")
    else:
        st.warning("Primero sube una imagen.")
