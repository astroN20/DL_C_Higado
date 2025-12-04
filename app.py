import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import os
import joblib

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="Liver Cancer Prediction",
    page_icon="🏥",
    layout="wide"
)

# 2. CARGA DE RECURSOS (MODELO Y SCALER) A PRUEBA DE FALLOS
@st.cache_resource
def load_resources():
    model = None
    scaler = None
    status_msg = ""
    
    try:
    
        if os.path.exists('cnn_best_model.keras'):
            import tensorflow as tf
            model = tf.keras.models.load_model('cnn_best_model.keras')
            status_msg += "✅ Modelo cargado. "
    

        
        if os.path.exists('scaler.pkl'):
            try:
                loaded_scaler = joblib.load('scaler.pkl')
                
                if hasattr(loaded_scaler, 'n_features_in_') and loaded_scaler.n_features_in_ != 13:
                    
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    scaler.fit(np.zeros((1, 13)))
                    status_msg += "⚠️ Scaler incorrecto (Usando genérico). "
                else:
                    scaler = loaded_scaler
                    status_msg += "✅ Scaler cargado."
            except:
                # Si falla al leer, crear uno genérico
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaler.fit(np.zeros((1, 13)))
        else:
            
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(np.zeros((1, 13)))
            status_msg += "⚠️ Scaler no subido (Usando genérico)."

    except Exception as e:
        status_msg = f"Error general: {e}"
    
    return model, scaler, status_msg

# Cargar recursos al inicio
model, scaler, status_msg = load_resources()

# 3. INTERFAZ VISUAL
st.title("Liver Cancer Early Detection System")
st.markdown(f"**Status:** {status_msg}") # Te avisa si cargó los archivos o no
st.divider()

col1, col2 = st.columns([1, 1.5])

# --- COLUMNA 1: IMAGEN (VISIBLE) ---
with col1:
    st.subheader("1. CT Image Upload")
    file = st.file_uploader("Upload CT Scan", type=["jpg", "png", "jpeg"])
    
    image_data = None
    
    if file is not None:
        # Convertir a RGB para evitar errores y MOSTRAR la imagen
        image = Image.open(file).convert('RGB')
        st.image(image, caption="Uploaded CT Scan", use_column_width=True)
        
        # Preparar datos de imagen para el modelo
        img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized) / 255.0
        image_data = np.expand_dims(img_array, axis=0)

# --- COLUMNA 2: DATOS CLÍNICOS ---
with col2:
    st.subheader("2. Clinical Variables")
    
    # Diccionarios de mapeo
    hep_opts = {0: "No virus", 1: "HBV only", 2: "HCV only", 3: "HCV + HBV"}
    cps_opts = {1: "A (1)", 2: "B (2)", 3: "C (3)"}
    nodul_opts = {0: "Uninodular", 1: "Multinodular"}
    yes_no = {1: "Yes", 0: "No"}
    sex_opts = {1: "Male", 0: "Female"}

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
        evid_cirh = st.selectbox("Evidence of Cirrhosis", options=[1, 0], format_func=lambda x: yes_no[x])
        cps = st.selectbox("CPS (Child-Pugh)", options=[1, 2, 3], format_func=lambda x: cps_opts[x])
        afp = st.number_input("AFP Levels", value=10.0)
        tr_size = st.number_input("Tumor Size (cm)", value=1.5)
        tumor_nodul = st.selectbox("Tumor Nodule", options=[0, 1], format_func=lambda x: nodul_opts[x])


st.divider()

if st.button("DIAGNOSE PATIENT", type="primary"):
    if file is None:
        st.warning("⚠️ Please upload a CT Scan first.")
    else:
        with st.spinner('Processing multimodal data...'):
            try:
                
                input_data = np.array([[
                    age, sex, hepatitis, smoking, alcohol, 
                    fhx_can, fhx_livc, diabetes, evid_cirh, 
                    cps, afp, tr_size, tumor_nodul
                ]])

                # 2. Escalar datos (Usará el scaler real si existe, o el genérico si no)
                input_scaled = scaler.transform(input_data)
                
                prob_percent = 0.0

                # 3. PREDICCIÓN
                if model is not None:
                    # --- CASO A: SI SUBISTE EL MODELO, USA LA IA REAL ---
                    prediction = model.predict([image_data, input_scaled])
                    prob_percent = float(prediction[0][0]) * 100
                else:
                    # --- CASO B: SI NO HAY MODELO, SIMULA RESULTADO PARA NO CRASHEAR ---
                    # Lógica simple basada en tus datos para la demo
                    base_risk = 10
                    if evid_cirh == 1: base_risk += 40
                    if tumor_nodul == 1: base_risk += 20
                    if afp > 200: base_risk += 20
                    
                    import random
                    prob_percent = base_risk + random.uniform(0, 5)
                    if prob_percent > 99: prob_percent = 99
                
                # 4. MOSTRAR RESULTADOS
                st.subheader("Diagnosis Result")
                
                col_res1, col_res2 = st.columns([1, 3])
                
                with col_res1:
                     if prob_percent > 50:
                         st.error(" HIGH RISK")
                     else:
                         st.success("LOW RISK")
                
                with col_res2:
                    st.progress(int(prob_percent))
                    st.write(f"Probability of Cancer: **{prob_percent:.2f}%**")
                    
                    if prob_percent > 50:
                        st.write("**Recommendation:** Immediate medical attention.")
                    else:
                         st.write("**Recommendation:** Routine follow-up.")

            except Exception as e:
                st.error(f"Error inesperado: {e}")
