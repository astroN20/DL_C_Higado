import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import joblib


st.set_page_config(
    page_title="Detección de Cáncer de Hígado",
    page_icon="🏥",
    layout="wide"
)

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

st.title("Sistema de Detección Temprana de Cáncer de Hígado")
st.markdown("Este sistema utiliza **Deep Learning Multimodal** integrando imágenes de TC y datos clínicos.")


col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Imagen de Tomografía (TC)")
    file = st.file_uploader("Cargar imagen (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    if file is not None:
        image = Image.open(file)
        st.image(image, caption="Imagen cargada", use_column_width=True)

with col2:
    st.subheader("2. Datos Clínicos del Paciente")
   
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("1. Edad", min_value=1, max_value=100, value=50)
        gender = st.selectbox("2. Género", options=[0, 1], format_func=lambda x: "Masculino" if x == 1 else "Femenino")
        bmi = st.number_input("3. Índice de Masa Corporal (BMI)", value=24.0)
        alcohol = st.selectbox("4. Consumo de Alcohol", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        smoking = st.selectbox("5. ¿Fuma?", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        diabetes = st.selectbox("6. Diabetes", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        hepatitis = st.selectbox("7. Hepatitis B/C", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

    with c2:
      
        cirrhosis = st.selectbox("8. Cirrosis", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        family_history = st.selectbox("9. Antecedentes Familiares", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        afp = st.number_input("10. Niveles AFP (ng/mL)", value=10.0)
        alt = st.number_input("11. Niveles ALT (U/L)", value=30.0)
        ast = st.number_input("12. Niveles AST (U/L)", value=30.0)
        tumor_size = st.number_input("13. Tamaño del Tumor (cm)", value=2.0)

    if st.button("Realizar Diagnóstico", type="primary"):
        if file is None:
            st.warning("⚠️ Por favor cargue una imagen primero.")
        else:
            with st.spinner('Analizando datos multimodales...'):
                try:
                   
                    img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
                    img_array = np.array(img) / 255.0
                    img_batch = np.expand_dims(img_array, axis=0)

                    datos_clinicos = np.array([[
                        age, gender, bmi, alcohol, smoking, diabetes, hepatitis,
                        cirrhosis, family_history, afp, alt, ast, tumor_size
                    ]]) 
                    
                 
                    prediction = model.predict([img_batch, datos_clinicos])
                    probabilidad = prediction[0][0] * 100
                    
                    st.divider()
                    st.subheader("Resultado del Análisis")
                    if probabilidad > 50:
                        st.error(f" **RIESGO ALTO DETECTADO**")
                        st.write(f"Probabilidad de Cáncer: **{probabilidad:.2f}%**")
                        st.progress(int(probabilidad))
                    else:
                        st.success(f" **BAJO RIESGO / SANO**")
                        st.write(f"Probabilidad de Cáncer: **{probabilidad:.2f}%**")
                        st.progress(int(probabilidad))
                        
                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")
                    st.warning("Revisa que el número de variables (13) coincida con tu entrenamiento.")

  
   
