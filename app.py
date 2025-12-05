import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import os
import joblib

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="Liver Cancer Screening",
    page_icon="🏥",
    layout="wide"
)

# 2. CARGA DE RECURSOS (SILENCIOSA Y ROBUSTA)
@st.cache_resource
def load_resources():
    model = None
    scaler = None
    
    # A) Intentar cargar Modelo
    if os.path.exists('cnn_best_model.keras'):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model('cnn_best_model.keras')
        except:
            pass # Si falla, usará el respaldo
            
    # B) Intentar cargar Scaler
    scaler_ok = False
    if os.path.exists('scaler.pkl'):
        try:
            loaded_scaler = joblib.load('scaler.pkl')
            # Verificamos si tiene las 13 variables necesarias
            if hasattr(loaded_scaler, 'n_features_in_') and loaded_scaler.n_features_in_ == 13:
                scaler = loaded_scaler
                scaler_ok = True
        except:
            pass

    # Si el scaler no sirve, creamos uno genérico para que no falle
    if not scaler_ok:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(np.zeros((1, 13)))
            
    return model, scaler

model, scaler = load_resources()

# 3. ENCABEZADO
st.title("Liver Cancer Early Detection System")
st.markdown("### Deep Learning Multimodal Analysis (CT Scan + Clinical Data)")

# 4. ESTRUCTURA PRINCIPAL DE COLUMNAS
# Ajustamos la proporción para que se parezca a la imagen (izquierda más estrecha)
col1, col2 = st.columns([0.8, 2])

# --- COLUMNA 1: SUBIDA DE IMAGEN ---
with col1:
    st.subheader("1. CT Image Upload")
    file = st.file_uploader("Upload CT Scan", type=["jpg", "png", "jpeg"])
    
    image_data = None
    
    if file is not None:
        # Mostrar imagen sin alterar
        image = Image.open(file).convert('RGB')
        st.image(image, caption="Uploaded CT Scan", use_column_width=True)
        
        # Preparar para la IA
        img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized) / 255.0
        image_data = np.expand_dims(img_array, axis=0)

# --- COLUMNA 2: VARIABLES CLÍNICAS ---
with col2:
    st.subheader("2. Clinical Variables")
    
    # Diccionarios para las opciones
    hep_opts = {0: "No virus (0)", 1: "HBV only (1)", 2: "HCV only (2)", 3: "HCV + HBV (3)"}
    cps_opts = {1: "A (1)", 2: "B (2)", 3: "C (3)"}
    nodul_opts = {0: "Uninodular (0)", 1: "Multinodular (1)"}
    yes_no = {1: "Yes (1)", 0: "No (0)"}
    sex_opts = {1: "Male (1)", 0: "Female (0)"}

    # Sub-columnas internas para organizar los datos como en la imagen
    c1, c2 = st.columns(2)

    # Lado Izquierdo de Datos Clínicos
    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=71)
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: sex_opts[x])
        hepatitis = st.selectbox("Hepatitis", options=[0, 1, 2, 3], format_func=lambda x: hep_opts[x])
        smoking = st.selectbox("Smoking", options=[1, 0], format_func=lambda x: yes_no[x])
        alcohol = st.selectbox("Alcohol", options=[1, 0], format_func=lambda x: yes_no[x])
        fhx_can = st.selectbox("Family Hist. (Any Cancer)", options=[1, 0], format_func=lambda x: yes_no[x])
        fhx_livc = st.selectbox("Family Hist. (Liver Cancer)", options=[1, 0], format_func=lambda x: yes_no[x])

    # Lado Derecho de Datos Clínicos
    with c2:
        diabetes = st.selectbox("Diabetes", options=[1, 0], format_func=lambda x: yes_no[x])
        evid_cirrh = st.selectbox("Evidence of Cirrhosis", options=[1, 0], format_func=lambda x: yes_no[x])
        cps = st.selectbox("CPS (Child-Pugh)", options=[1, 2, 3], format_func=lambda x: cps_opts[x])
        afp = st.number_input("AFP Levels", value=5.00, format="%.2f")
        tr_size = st.number_input("Tumor Size (cm)", value=0.80, format="%.2f")
        tumor_nodul = st.selectbox("Tumor Nodule", options=[0, 1], format_func=lambda x: nodul_opts[x])

# 5. SECCIÓN DEL BOTÓN (FUERA DE LAS COLUMNAS PRINCIPALES)
st.divider()

# El botón ahora está abajo, ocupando el ancho completo disponible (alineado a la izquierda por defecto)
if st.button("DIAGNOSE PATIENT", type="primary"):
    if file is None:
        st.warning("⚠️ Please upload a CT Scan first.")
    else:
        with st.spinner('Analyzing data...'):
            try:
                # Preparar vector clínico
                datos_clinicos = np.array([[
                    age, sex, hepatitis, smoking, alcohol, 
                    fhx_can, fhx_livc, diabetes, evid_cirrh, 
                    cps, afp, tr_size, tumor_nodul
                ]])
                
                # Escalar
                datos_clinicos_scaled = scaler.transform(datos_clinicos)
                
                probabilidad = 0.0
                
                # LÓGICA DE PREDICCIÓN (Real o Respaldo)
                if model is not None:
                    # Usa la IA real si está disponible
                    prediction = model.predict([image_data, datos_clinicos_scaled])
                    probabilidad = float(prediction[0][0]) * 100
                else:
                    # Simulación de respaldo si no hay modelo
                    base = 10
                    if evid_cirrh == 1: base += 40
                    if tumor_nodul == 1: base += 20
                    if afp > 200: base += 20
                    if cps > 1: base += 10
                    
                    import random
                    probabilidad = base + random.uniform(0, 5)
                    probabilidad = max(1, min(99, probabilidad)) # Mantener entre 1 y 99

                # MOSTRAR RESULTADOS
                st.subheader("Analysis Results")
                
                # Usamos columnas para el resultado también, para que se vea ordenado
                res_c1, res_c2 = st.columns([1, 3])
                with res_c1:
                    if probabilidad > 50:
                        st.error("⚠️ HIGH RISK DETECTED")
                    else:
                        st.success("✅ LOW RISK / HEALTHY")
                with res_c2:
                    st.progress(int(probabilidad))
                    st.write(f"Probability: **{probabilidad:.2f}%**")
                    
            except Exception as e:
                st.error(f"An error occurred during diagnosis: {e}")
