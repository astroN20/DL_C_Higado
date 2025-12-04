import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import joblib

# Configuración de la página
st.set_page_config(
    page_title="Liver Cancer Screening",
    page_icon="🏥",
    layout="wide"
)

# Cargar recursos
@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model('cnn_best_model.keras')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error al cargar el modelo o el scaler: {e}")
        st.stop()

model, scaler = load_resources()

st.title("Liver Cancer Early Detection System")
st.markdown("This system uses **Deep Learning Multimodal** by integrating CT images and clinical data.")

col1, col2 = st.columns([1, 1.5])

# --- COLUMNA 1: IMAGEN ---
with col1:
    st.subheader("1. Image Computed Tomography (CT)")
    file = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "png", "jpeg"])
    image = None
    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, caption="Image uploaded", use_column_width=True)

# --- COLUMNA 2: DATOS CLÍNICOS EXACTOS ---
with col2:
    st.subheader("2. Patient Clinical Data")
    
    # Diccionarios para etiquetas (según tus capturas)
    hep_dict = {0: "No virus", 1: "HBV only", 2: "HCV only", 3: "HCV and HBV"}
    cps_dict = {1: "A (1)", 2: "B (2)", 3: "C (3)"}
    nodul_dict = {0: "Uninodular", 1: "Multinodular"}
    
    c1, c2 = st.columns(2)

    with c1:
        # 1. Age
        age = st.number_input("1. Age", min_value=1, max_value=100, value=40)
        
        # 2. Sex (Asumiendo 0=Female, 1=Male o viceversa, verifica tu entrenamiento)
        sex = st.selectbox("2. Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        
        # 3. Hepatitis (SEGÚN TU IMAGEN: 0, 1, 2, 3)
        hepatitis = st.selectbox("3. Hepatitis Status", options=[0, 1, 2, 3], format_func=lambda x: hep_dict[x])
        
        # 4. Smoking
        smoking = st.selectbox("4. Smoking", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        
        # 5. Alcohol
        alcohol = st.selectbox("5. Alcohol", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        
        # 6. Family History Cancer (fhx_can)
        fhx_can = st.selectbox("6. Family Hist. (Any Cancer)", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

        # 7. Family History Liver Cancer (fhx_livc)
        fhx_livc = st.selectbox("7. Family Hist. (Liver Cancer)", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    with c2:
        # 8. Diabetes
        diabetes = st.selectbox("8. Diabetes", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        
        # 9. Evidence of Cirrhosis
        evid_cirrh = st.selectbox("9. Evidence of Cirrhosis", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        
        # 10. CPS (SEGÚN TU IMAGEN: 1, 2, 3)
        cps = st.selectbox("10. Child-Pugh Score (CPS)", options=[1, 2, 3], format_func=lambda x: cps_dict[x])
        
        # 11. AFP
        afp = st.number_input("11. AFP Levels (ng/mL)", value=0.0)
        
        # 12. Tr_Size (Tumor Size)
        tr_size = st.number_input("12. Tumor Size (cm)", value=0.0)
        
        # 13. Tumor Nodule (SEGÚN TU IMAGEN: 0, 1)
        tumor_nodul = st.selectbox("13. Tumor Nodule", options=[0, 1], format_func=lambda x: nodul_dict[x])

    # --- BOTÓN DE PREDICCIÓN ---
    if st.button("Perform Diagnosis", type="primary"):
        if file is None or image is None:
            st.warning("⚠️ Please upload an image first.")
        else:
            with st.spinner('Analyzing multimodal data...'):
                try:
                    # 1. Procesamiento de Imagen
                    img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
                    img_array = np.array(img_resized) / 255.0
                    img_batch = np.expand_dims(img_array, axis=0)

                    # 2. Procesamiento de Datos Clínicos
                    # ORDEN EXACTO DE LA IMAGEN AMARILLA:
                    # [age, Sex, hepatitis, Smoking, Alcohol, fhx_can, fhx_livc, Diabetes, Evidence_of_cirh, CPS, AFP, Tr_Size, tumor_nodul]
                    
                    datos_clinicos = np.array([[
                        age, sex, hepatitis, smoking, alcohol, 
                        fhx_can, fhx_livc, diabetes, evid_cirrh, 
                        cps, afp, tr_size, tumor_nodul
                    ]])
                    
                    # 3. Escalar y Predecir
                    datos_clinicos_scaled = scaler.transform(datos_clinicos)
                    prediction = model.predict([img_batch, datos_clinicos_scaled])
                    probabilidad = prediction[0][0] * 100
                    
                    # 4. Resultados
                    st.divider()
                    st.subheader("Analysis Results")
                    if probabilidad > 50:
                        st.error(f"**HIGH RISK DETECTED**")
                        st.write(f"Probabilidad de Cáncer: **{probabilidad:.2f}%**")
                        st.progress(int(probabilidad))
                    else:
                        st.success(f"**LOW RISK / HEALTHY**")
                        st.write(f"Probabilidad de Cáncer: **{probabilidad:.2f}%**")
                        st.progress(int(probabilidad))
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.warning("Ensure your Scaler was trained with exactly these 13 variables in this order.")
