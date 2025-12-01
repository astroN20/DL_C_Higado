import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import joblib
import os
import gdown

st.set_page_config(page_title="Detección de Cáncer de Hígado", layout="wide")

@st.cache_resource
def load_resources():
    model_path = 'cnn_best_model.keras'
    scaler_path = 'scaler.pkl'
    
 
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        st.warning("Detectado archivo de modelo incompleto (Git LFS pointer). Descargando original...")
   
        file_id = 'PEGA_AQUI_TU_ID_DE_DRIVE_SIN_BORRAR_COMILLAS' 
        
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(url, model_path, quiet=False)
            st.success("✅ Modelo descargado correctamente.")
        except Exception as e:
            st.error(f"Error al descargar: {e}")
            st.stop()


    try:
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error crítico al cargar el modelo: {e}")
        st.stop()


model, scaler = load_resources()


st.title("🏥 Detección Temprana de Cáncer de Hígado")
st.write("Sistema de Deep Learning Multimodal (Imágenes TC + Datos Clínicos)")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Imagen TC")
    file = st.file_uploader("Subir imagen", type=["jpg", "png", "jpeg"])
    if file:
        image = Image.open(file)
        st.image(image, use_column_width=True)

with col2:
    st.subheader("2. Datos Clínicos")
 
    age = st.number_input("Edad", 1, 100, 50)
    
    if st.button("Predecir", type="primary"):
        if file is None:
            st.error("Falta la imagen")
        else:
            with st.spinner('Procesando...'):
                
                img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
                img_array = np.array(img) / 255.0
                img_batch = np.expand_dims(img_array, axis=0)
                
           
                datos = np.array([[age]]) 
                
                try:
                    pred = model.predict([img_batch, datos]) 
                    prob = pred[0][0] * 100
                    
                    st.divider()
                    if prob > 50:
                        st.error(f"Riesgo Alto: {prob:.2f}%")
                    else:
                        st.success(f"Riesgo Bajo: {prob:.2f}%")
                except Exception as e:
                    st.warning("El modelo cargó bien, pero falló la predicción por las variables.")
                    st.error(f"Detalle: {e}")
       
