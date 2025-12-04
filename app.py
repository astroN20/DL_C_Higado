import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import joblib

# Configuración de la página (Debe ser la primera instrucción de Streamlit)
st.set_page_config(
    page_title="Liver Cancer Screening",
    page_icon="🏥",
    layout="wide"
)

# Cargar recursos (Modelo y Scaler) una sola vez
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

# Título y Descripción
st.title("Liver Cancer Early Detection System")
st.markdown("This system uses **Deep Learning Multimodal** by integrating CT images and clinical data.")

# Crear columnas
col1, col2 = st.columns([1, 1.5])

# --- COLUMNA 1: IMAGEN ---
with col1:
    st.subheader("1. Image Computed Tomography (CT)")
    file = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    # Variable para guardar la imagen procesada
    image = None

    if file is not None:
        # CORRECCIÓN 1: Convertir a RGB para evitar error con PNG
        image = Image.open(file).convert('RGB')
        st.image(image, caption="Image uploaded", use_column_width=True)

# --- COLUMNA 2: DATOS CLÍNICOS ---
with col2:
    st.subheader("2. Patient Clinical Data")
    
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("1. Age", min_value=1, max_value=100, value=40)
        gender = st.selectbox("2. Gender", options=[0, 1], format_func=lambda x: "Masculino" if x == 1 else "Femenino")
        bmi = st.number_input("3. Body Mass Index (BMI)", value=0.0)
        alcohol = st.selectbox("4. Alcohol Consumption", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        smoking = st.selectbox("5. Smoke?", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        diabetes = st.selectbox("6. Diabetes", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        hepatitis = st.selectbox("7. Hepatitis B/C", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

    with c2:
        cirrhosis = st.selectbox("8. Cirrhosis", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        family_history = st.selectbox("9. Family History", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        afp = st.number_input("10. AFP Levels (ng/mL)", value=0.0)
        alt = st.number_input("11. ALT Levels (U/L)", value=0.0)
        ast = st.number_input("12. AST Levels (U/L)", value=0.0)
        tumor_size = st.number_input("13. Tamaño del Tumor (cm)", value=0.0)

    # --- BOTÓN DE PREDICCIÓN ---
    # CORRECCIÓN 2: Indentación correcta del bloque del botón
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
                    datos_clinicos = np.array([[
                        age, gender, bmi, alcohol, smoking, diabetes, hepatitis,
                        cirrhosis, family_history, afp, alt, ast, tumor_size
                    ]]) 
                    
                    datos_clinicos_scaled = scaler.transform(datos_clinicos)
                    
                  
                    prediction = model.predict([img_batch, datos_clinicos_scaled])
                    probabilidad = prediction[0][0] * 100
                    
                   
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
                    st.warning("Check that the number of variables (13) matches your training.")
  
   
