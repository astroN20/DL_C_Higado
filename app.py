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

    model = tf.keras.models.load_model('mi_modelo_cnn.keras') 
 
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_resources()
except Exception as e:
    st.error(f"Error al cargar archivos: {e}. Verifica que 'mi_modelo_cnn.keras' y 'scaler.pkl' estén en la carpeta.")
    st.stop()


st.title("🏥 Sistema de Detección Temprana de Cáncer de Hígado")
st.markdown("""
Este sistema utiliza **Deep Learning Multimodal** integrando imágenes de TC y datos clínicos 
para estimar la probabilidad de carcinoma hepatocelular.
""")


col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Imagen de Tomografía (TC)")
    file = st.file_uploader("Cargar imagen (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    if file is not None:
        image = Image.open(file)
        st.image(image, caption="Imagen cargada", use_column_width=True)

with col2:
    st.subheader("2. Datos Clínicos del Paciente")
    st.info("Por favor ingrese los valores clínicos.")

    c1, c2 = st.columns(2)
    
    with c1:
        age = st.number_input("Edad", min_value=1, max_value=100, value=50)
        gender = st.selectbox("Género", options=[0, 1], format_func=lambda x: "Masculino" if x == 1 else "Femenino")
        hepatitis = st.selectbox("Hepatitis", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        diabetes = st.selectbox("Diabetes", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        alcohol = st.selectbox("Consumo de Alcohol", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

    with c2:
        cirrhosis = st.selectbox("Cirrosis", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        afp = st.number_input("AFP (Alpha-fetoprotein)", value=10.0)
        tumor_size = st.number_input("Tamaño del tumor (cm)", value=2.0)

    if st.button("Realizar Diagnóstico", type="primary"):
        if file is None:
            st.warning(" Por favor cargue una imagen primero.")
        else:
            with st.spinner('Analizando datos multimodales...'):
                try:
                 
                    img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
                    img_array = np.array(img)
                    
                
                    img_array = img_array / 255.0  
                    
             
                    img_batch = np.expand_dims(img_array, axis=0)

            
                    datos_clinicos = np.array([[age, gender, hepatitis, diabetes, alcohol, cirrhosis, afp, tumor_size]]) 
                    
                
                    datos_clinicos_scaled = datos_clinicos 

                    
                    prediction = model.predict([img_batch, datos_clinicos_scaled])
                    
                  
                    score = prediction[0][0]
                    probabilidad = score * 100
                    
                    st.divider()
                    if score > 0.5:
                        st.error(f"🚨 **Resultado: ALTO RIESGO DE CÁNCER (Clase Positiva)**")
                        st.write(f"Probabilidad estimada: **{probabilidad:.2f}%**")
                    else:
                        st.success(f"✅ **Resultado: BAJO RIESGO (Clase Negativa)**")
                        st.write(f"Probabilidad estimada: **{probabilidad:.2f}%**")
                        
                except Exception as e:
                    st.error(f"Error en el procesamiento: {e}")
                    st.info("Consejo: Verifica que el número de variables clínicas coincida con el entrenamiento.")