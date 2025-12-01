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

st.title("🏥 Sistema de Detección Temprana de Cáncer de Hígado")
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

    age = st.number_input("Edad", min_value=1, max_value=100, value=50)
    gender = st.selectbox("Género", options=[0, 1], format_func=lambda x: "Masculino" if x == 1 else "Femenino")


  
    if st.button("Realizar Diagnóstico", type="primary"):
        if file is None:
            st.warning("Por favor cargue una imagen primero.")
        else:
            with st.spinner('Analizando datos...'):
                try:
                  
                    img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
                    img_array = np.array(img) / 255.0
                    img_batch = np.expand_dims(img_array, axis=0)

              
                    datos_clinicos = np.array([[age, gender]]) 
                    
                
                    prediction = model.predict([img_batch, datos_clinicos])
                    
                    probabilidad = prediction[0][0] * 100
                    
                    st.divider()
                    if probabilidad > 50:
                        st.error(f"RIESGO ALTO: {probabilidad:.2f}%")
                    else:
                        st.success(f" RIESGO BAJO: {probabilidad:.2f}%")
                        
                except Exception as e:
                    st.error(f"Ocurrió un error en la predicción: {e}")
