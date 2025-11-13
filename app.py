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

st.title("Detección de Cáncer de Hígado — Deep Learning")
st.markdown("Carga una **imagen de TC** y los **datos clínicos del paciente** .")

# --- Entrada de imagen ---
img_file = st.file_uploader(" Subir imagen (PNG o JPG)", type=["png", "jpg", "jpeg"])
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

# --- Recolectar los datos del usuario ---
# --- Entradas clínicas ---
st.subheader(" Datos clínicos del paciente")

# 1️⃣ Primera fila
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Edad", min_value=0, max_value=120, value=50)
with col2:
    sex = st.selectbox("Sexo", ["Male", "Female"])
    sex_code = 1 if sex == "Male" else 2

# 2️⃣ Segunda fila
col1, col2 = st.columns(2)
with col1:
    hepatitis = st.selectbox("Tipo de Hepatitis", ["No virus", "HBV only", "HCV only", "HCV and HBV"])
    hepatitis_code = {"No virus": 0, "HBV only": 1, "HCV only": 2, "HCV and HBV": 3}[hepatitis]
with col2:
    smoking = st.selectbox("¿Fuma?", ["No", "Yes"])
    smoking_code = 1 if smoking == "Yes" else 0

# 3️⃣ Tercera fila
col1, col2 = st.columns(2)
with col1:
    alcohol = st.selectbox("¿Consume alcohol?", ["No", "Yes"])
    alcohol_code = 1 if alcohol == "Yes" else 0
with col2:
    diabetes = st.selectbox("¿Tiene diabetes?", ["No", "Yes"])
    diabetes_code = 1 if diabetes == "Yes" else 0

# 4️⃣ Cuarta fila
col1, col2 = st.columns(2)
with col1:
    fhx_can = st.selectbox("Historial familiar de cáncer", ["No", "Yes"])
    fhx_can_code = 1 if fhx_can == "Yes" else 0
with col2:
    fhx_livc = st.selectbox("Historial familiar de cáncer de hígado", ["No", "Yes"])
    fhx_livc_code = 1 if fhx_livc == "Yes" else 0

# 5️⃣ Quinta fila
col1, col2 = st.columns(2)
with col1:
    evidence_of_cirh = st.selectbox("Evidencia de cirrosis", ["No", "Yes"])
    evidence_of_cirh_code = 1 if evidence_of_cirh == "Yes" else 0
with col2:
    cps = st.selectbox("Child-Pugh Score (CPS)", ["A", "B", "C"])
    cps_code = {"A": 1, "B": 2, "C": 3}[cps]

# 6️⃣ Sexta fila
col1, col2 = st.columns(2)
with col1:
    afp = st.number_input("AFP (Alpha-fetoprotein) (ng/ml)", min_value=0.0, value=10.0)
with col2:
    tr_size = st.number_input("Tamaño del tumor (cm)", min_value=0.0, value=2.0)

# 7️⃣ Séptima fila
tumor_nodul = st.selectbox("Tumor nodularidad", ["Uninodular", "Multinodular"])
tumor_nodul_code = 0 if tumor_nodul == "Uninodular" else 1

# --- Crear el DataFrame ---
clin_data = {
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
clin_df = pd.DataFrame([clin_data])


# --- Construir el diccionario de datos clínicos ---
clin_data = {
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

# --- Convertir a DataFrame ---
clin_df = pd.DataFrame([clin_data])



# Crear columnas faltantes con 0
for col in scaler.feature_names_in_:
    if col not in clin_df.columns:
        clin_df[col] = 0

# Reordenar exactamente igual que el scaler
clin_df = clin_df[scaler.feature_names_in_]

# Escalar
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
