import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import joblib
import os # Necesario para verificar archivos

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="Liver Cancer Screening",
    page_icon="🏥",
    layout="wide"
)

# --- BLOQUE DE DIAGNÓSTICO (Para ver qué archivos tienes realmente) ---
st.markdown("### 🔍 Diagnóstico de Archivos (Debug Mode)")
st.write(f"Carpeta actual: `{os.getcwd()}`")
archivos_encontrados = os.listdir()
st.write("Archivos detectados en la carpeta:", archivos_encontrados)

if 'cnn_best_model.keras' not in archivos_encontrados:
    st.error("❌ ERROR CRÍTICO: No se encuentra el archivo 'cnn_best_model.keras'. Por favor súbelo.")
else:
    st.success("✅ Modelo 'cnn_best_model.keras' detectado.")
st.divider()
# ---------------------------------------------------------------------

# 2. CARGA DE RECURSOS (Modelo y Scaler)
@st.cache_resource
def load_resources():
    try:
        # A) Cargar Modelo
        if not os.path.exists('cnn_best_model.keras'):
            st.stop() # Se detiene si no hay modelo, el mensaje de error ya se mostró arriba
        
        model = tf.keras.models.load_model('cnn_best_model.keras')
        
        # B) Cargar Scaler (Con protección "Anti-Crash")
        scaler = None
        if os.path.exists('scaler.pkl'):
            try:
                loaded_scaler = joblib.load('scaler.pkl')
                # Verificamos si tiene las 13 variables necesarias
                if hasattr(loaded_scaler, 'n_features_in_') and loaded_scaler.n_features_in_ != 13:
                    st.warning(f"⚠️ El scaler.pkl tiene {loaded_scaler.n_features_in_} variables, pero necesitamos 13. Usando scaler genérico temporal.")
                    # Crear scaler temporal
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    scaler.fit(np.zeros((1, 13)))
                else:
                    scaler = loaded_scaler
            except Exception as e:
                st.warning(f"⚠️ Error leyendo scaler.pkl: {e}. Usando genérico.")
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaler.fit(np.zeros((1, 13)))
        else:
            st.warning("⚠️ No existe 'scaler.pkl'. Usando scaler genérico temporal.")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(np.zeros((1, 13)))

        return model, scaler

    except Exception as e:
        st.error(f"Error fatal cargando recursos: {e}")
        st.stop()

model, scaler = load_resources()

# 3. INTERFAZ DE USUARIO
st.title("Liver Cancer Early Detection System")
st.markdown("This system uses **Deep Learning Multimodal** by integrating CT images and clinical data.")

col1, col2 = st.columns([1, 1.5])

# --- COLUMNA 1: IMAGEN ---
with col1:
    st.subheader("1. Image Computed Tomography (CT)")
    file = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "png", "jpeg"])
    image = None
    if file is not None:
        # Corrección para PNG (RGBA -> RGB)
        image = Image.open(file).convert('RGB')
        st.image(image, caption="Image uploaded", use_column_width=True)

# --- COLUMNA 2: DATOS CLÍNICOS ---
with col2:
    st.subheader("2. Patient Clinical Data")
    
    # Diccionarios para etiquetas
    hep_dict = {0: "No virus", 1: "HBV only", 2: "HCV only", 3: "HCV and HBV"}
    cps_dict = {1: "A (1)", 2: "B (2)", 3: "C (3)"}
    nodul_dict = {0: "Uninodular", 1: "Multinodular"}
    
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("1. Age", min_value=1, max_value=100, value=40)
        sex = st.selectbox("2. Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        hepatitis = st.selectbox("3. Hepatitis Status", options=[0, 1, 2, 3], format_func=lambda x: hep_dict[x])
        smoking = st.selectbox("4. Smoking", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        alcohol = st.selectbox("5. Alcohol", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        fhx_can = st.selectbox("6. Family Hist. (Any Cancer)", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        fhx_livc = st.selectbox("7. Family Hist. (Liver Cancer)", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    with c2:
        diabetes = st.selectbox("8. Diabetes", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        evid_cirrh = st.selectbox("9. Evidence of Cirrhosis", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        cps = st.selectbox("10. Child-Pugh Score (CPS)", options=[1, 2, 3], format_func=lambda x: cps_dict[x])
        afp = st.number_input("11. AFP Levels (ng/mL)", value=0.0)
        tr_size = st.number_input("12. Tumor Size (cm)", value=0.0)
        tumor_nodul = st.selectbox("13. Tumor Nodule", options=[0, 1], format_func=lambda x: nodul_dict[x])

    # --- BOTÓN DE PREDICCIÓN ---
    if st.button("Perform Diagnosis", type="primary"):
        if file is None or image is None:
            st.warning("⚠️ Please upload an image first.")
        else:
            with st.spinner('Analyzing multimodal data...'):
                try:
                    # A. Procesar Imagen
                    img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
                    img_array = np.array(img_resized) / 255.0
                    img_batch = np.expand_dims(img_array, axis=0)

                    # B. Procesar Datos Clínicos (Orden estricto de 13 variables)
                    datos_clinicos = np.array([[
                        age, sex, hepatitis, smoking, alcohol, 
                        fhx_can, fhx_livc, diabetes, evid_cirrh, 
                        cps, afp, tr_size, tumor_nodul
                    ]])
                    
                    # C. Escalar (Usando el scaler real o el temporal)
                    datos_clinicos_scaled = scaler.transform(datos_clinicos)
                    
                    # D. Predecir
                    prediction = model.predict([img_batch, datos_clinicos_scaled])
                    probabilidad = prediction[0][0] * 100
                    
                    # E. Mostrar Resultados
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
                    st.error(f"Ocurrió un error en la predicción: {e}")
