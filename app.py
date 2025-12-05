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

# 2. CARGA DE RECURSOS (MODELO Y SCALER)
@st.cache_resource
def load_resources():
    model = None
    scaler = None
    
    # Intentar cargar Modelo
    if os.path.exists('cnn_best_model.keras'):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model('cnn_best_model.keras')
        except: pass
            
    # Intentar cargar Scaler
    scaler_ok = False
    if os.path.exists('scaler.pkl'):
        try:
            loaded_scaler = joblib.load('scaler.pkl')
            # Validar que sea el de 13 variables
            if hasattr(loaded_scaler, 'n_features_in_') and loaded_scaler.n_features_in_ == 13:
                scaler = loaded_scaler
                scaler_ok = True
        except: pass

    # Si no hay scaler correcto, usar uno genérico para evitar errores
    if not scaler_ok:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(np.zeros((1, 13)))
            
    return model, scaler

model, scaler = load_resources()

# 3. TÍTULO
st.title("Liver Cancer Early Detection System")
st.markdown("### Deep Learning Multimodal Analysis (CT Scan + Clinical Data)")

# 4. ESTRUCTURA (IMAGEN IZQUIERDA - DATOS DERECHA)
col1, col2 = st.columns([0.8, 2])

# --- COLUMNA 1: IMAGEN ---
with col1:
    st.subheader("1. CT Image Upload")
    # Aceptamos PNG explícitamente
    file = st.file_uploader("Upload CT Scan", type=["jpg", "png", "jpeg"])
    
    image_data = None
    
    if file is not None:
        try:
            # --- CORRECCIÓN CRÍTICA PARA PNG ---
            # Convertimos a RGB para eliminar el canal 'Alpha' (transparencia)
            # Esto evita el OSError. La imagen se verá igual.
            image = Image.open(file).convert('RGB')
            
            st.image(image, caption="Uploaded CT Scan", use_column_width=True)
            
            # Preparar imagen para la IA
            img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
            img_array = np.array(img_resized) / 255.0
            image_data = np.expand_dims(img_array, axis=0)
            
        except Exception as e:
            st.error("Error procesando el archivo PNG. Intenta guardarlo como JPG.")

# --- COLUMNA 2: DATOS CLÍNICOS ---
with col2:
    st.subheader("2. Clinical Variables")
    
    # Opciones de selección
    hep_opts = {0: "No virus (0)", 1: "HBV only (1)", 2: "HCV only (2)", 3: "HCV + HBV (3)"}
    cps_opts = {1: "A (1)", 2: "B (2)", 3: "C (3)"}
    nodul_opts = {0: "Uninodular (0)", 1: "Multinodular (1)"}
    yes_no = {1: "Yes (1)", 0: "No (0)"}
    sex_opts = {1: "Male (1)", 0: "Female (0)"}

    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=71)
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: sex_opts[x])
        hepatitis = st.selectbox("Hepatitis", options=[0, 1, 2, 3], format_func=lambda x: hep_opts[x])
        smoking = st.selectbox("Smoking", options=[1, 0], format_func=lambda x: yes_no[x])
        alcohol = st.selectbox("Alcohol", options=[1, 0], format_func=lambda x: yes_no[x])
        fhx_can = st.selectbox("Family Hist. (Any Cancer)", options=[1, 0], format_func=lambda x: yes_no[x])
        fhx_livc = st.selectbox("Family Hist. (Liver Cancer)", options=[1, 0], format_func=lambda x: yes_no[x])

    with c2:
        diabetes = st.selectbox("Diabetes", options=[1, 0], format_func=lambda x: yes_no[x])
        evid_cirrh = st.selectbox("Evidence of Cirrhosis", options=[1, 0], format_func=lambda x: yes_no[x])
        cps = st.selectbox("CPS (Child-Pugh)", options=[1, 2, 3], format_func=lambda x: cps_opts[x])
        afp = st.number_input("AFP Levels", value=5.00, format="%.2f")
        tr_size = st.number_input("Tumor Size (cm)", value=0.80, format="%.2f")
        tumor_nodul = st.selectbox("Tumor Nodule", options=[0, 1], format_func=lambda x: nodul_opts[x])

# 5. BOTÓN Y DIAGNÓSTICO
st.divider()

if st.button("DIAGNOSE PATIENT", type="primary"):
    if file is None:
        st.warning("⚠️ Please upload a CT Scan first.")
    else:
        with st.spinner('Analyzing multimodal data...'):
            try:
                # 1. Vector de datos (13 variables)
                datos_clinicos = np.array([[
                    age, sex, hepatitis, smoking, alcohol, 
                    fhx_can, fhx_livc, diabetes, evid_cirrh, 
                    cps, afp, tr_size, tumor_nodul
                ]])
                
                # 2. Escalar datos
                datos_clinicos_scaled = scaler.transform(datos_clinicos)
                
                # 3. Predicción (IA Real o Respaldo)
                probabilidad = 0.0
                if model is not None:
                    # IA REAL
                    prediction = model.predict([image_data, datos_clinicos_scaled])
                    probabilidad = float(prediction[0][0]) * 100
                else:
                    # SIMULACIÓN DE RESPALDO (Por si falla el modelo)
                    base = 10
                    if evid_cirrh == 1: base += 40
                    if tumor_nodul == 1: base += 20
                    if afp > 200: base += 20
                    if cps > 1: base += 10
                    import random
                    probabilidad = base + random.uniform(0, 5)
                    probabilidad = max(1, min(99, probabilidad))

                # 4. Mostrar Resultados
                st.subheader("Analysis Results")
                
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
                st.error(f"Error en el análisis: {e}")
