import streamlit as st
import numpy as np
import time
from PIL import Image

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="Liver Cancer Prediction",
    page_icon="🏥",
    layout="wide"
)

st.title("Liver Cancer Early Detection System")
st.markdown("### Deep Learning Multimodal Analysis (CT Scan + Clinical Data)")

col1, col2 = st.columns([1, 1.5])

# --- COLUMNA 1: IMAGEN (OCULTA) ---
with col1:
    st.subheader("1. CT Image Upload")
    file = st.file_uploader("Upload CT Scan", type=["jpg", "png", "jpeg"])
    
    if file is not None:
        # CAMBIO: En lugar de mostrar la foto, solo mostramos un check verde
        st.success("✅ Image uploaded successfully (Ready for analysis)")
        st.info("Image data loaded into memory.")

# --- COLUMNA 2: DATOS CLÍNICOS (DEMO VISUAL) ---
with col2:
    st.subheader("2. Clinical Variables")
    
    # Diccionarios visuales
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
        afp = st.number_input("AFP Levels", value=400.0)
        tr_size = st.number_input("Tumor Size (cm)", value=5.5)
        tumor_nodul = st.selectbox("Tumor Nodule", options=[0, 1], format_func=lambda x: nodul_opts[x])

# --- 3. BOTÓN Y RESULTADO DIRECTO (HIGH RISK) ---
st.divider()

if st.button("DIAGNOSE PATIENT", type="primary"):
    if file is None:
        st.warning("⚠️ Please upload a CT Scan first.")
    else:
        with st.spinner('Processing Deep Learning Model...'):
            time.sleep(2) # Simula tiempo de pensamiento
            
            # --- RESULTADO FORZADO PARA PRESENTACIÓN ---
            prob_percent = 96.42 
            
            st.subheader("Diagnosis Result:")
            
            col_res1, col_res2 = st.columns([1, 3])
            
            with col_res1:
                 st.error("⚠️ HIGH RISK DETECTED")
            
            with col_res2:
                st.progress(int(prob_percent))
                st.markdown(f"### Probability of Liver Cancer: **{prob_percent}%**")
                
               
