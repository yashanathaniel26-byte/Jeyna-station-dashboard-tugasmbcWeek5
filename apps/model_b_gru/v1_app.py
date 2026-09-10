import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import tensorflow as tf
import sys

# Memastikan modul utils bisa diakses
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.preprocessing import prepare_sequence_from_dict, prepare_sequence_from_df, inverse_scale_temperature

# Konfigurasi Halaman
st.set_page_config(page_title="Model B (GRU) - v1.0 Baseline", layout="wide")

st.title("🌬️ Model B: GRU Weather Indicator (v1.0)")
st.write("Versi 1.0 - Baseline Inference Univariate (T degC) tanpa optimasi (No Caching)")

# --- SIDEBAR ---
st.sidebar.header("Navigation & Status")
st.sidebar.markdown("**Active Station Focus:**")
st.sidebar.selectbox("Pilih Stasiun", ["Jena Central (Default)"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Metrics (Baseline)")
st.sidebar.text("Model: GRU Config 1 (L1)")
st.sidebar.text("Sequence Length: 144")
st.sidebar.text("Input Features: 1 (T degC)")
st.sidebar.text("MAE: 0.0142")
st.sidebar.text("RMSE: 0.0221")

# --- MAIN WORKSPACE ---
st.subheader("Data Input")
input_mode = st.radio("Pilih Mode Input:", ["Manual Input", "CSV Upload"])

input_sequence = None

if input_mode == "Manual Input":
    st.write("Masukkan nilai suhu saat ini (akan diduplikasi menjadi 144 timesteps):")
    manual_data = {}
    manual_data['T (degC)'] = st.number_input("Suhu T (degC)", value=15.0, key="gru_temp")
            
    if st.button("Generate Sequence dari Input Manual"):
        input_sequence = prepare_sequence_from_dict(manual_data)
        st.success(f"Sequence terbentuk! Shape: {input_sequence.shape} (Batch, Timesteps, Features)")
        
elif input_mode == "CSV Upload":
    uploaded_file = st.file_uploader("Unggah file sample_jena.csv (minimal 144 baris)", type=['csv'], key="gru_csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview Data (Tail):", df[['T (degC)']].tail())
        if st.button("Generate Sequence dari CSV"):
            try:
                input_sequence = prepare_sequence_from_df(df)
                st.success(f"Sequence terbentuk! Shape: {input_sequence.shape}")
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.subheader("Inference & Prediction")

if st.button("Predict Weather Status", type="primary"):
    if input_sequence is None:
        st.warning("Silakan generate sequence input terlebih dahulu di atas!")
    else:
        with st.spinner("Memuat model GRU tanpa cache (Simulasi Load Lambat)..."):
            start_load = time.time()
            
            # Load model TANPA @st.cache_resource
            model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'gru_config1.h5')
            try:
                model = tf.keras.models.load_model(model_path)
                load_time = time.time() - start_load
                st.info(f"Model Load Time: {load_time:.2f} detik")
                
                # Inference
                start_infer = time.time()
                prediction = model.predict(input_sequence)
                infer_time = time.time() - start_infer
                st.info(f"Inference Latency: {infer_time*1000:.2f} ms")
                
                scaled_predicted_value = float(prediction[0][0])
                actual_temp = inverse_scale_temperature(scaled_predicted_value)
                
                st.success("✅ Prediksi Selesai")
                st.write(f"Suhu Terprediksi: {actual_temp:.2f} °C")
                
                # Logika badge berdasarkan prediksi Suhu (karena GRU dilatih univariat Suhu)
                status = "NORMAL (SUHU SEJUK)"
                color = "green"
                if actual_temp > 30.0:
                    status = "WASPADA (SUHU PANAS EKSTREM)"
                    color = "red"
                elif actual_temp < 5.0:
                    status = "WASPADA (RAWAN BEKU / FROST)"
                    color = "blue"
                    
                st.markdown(f"### Status Prediksi Iklim:")
                st.markdown(f"<h2 style='text-align: center; color: white; background-color: {color}; padding: 10px; border-radius: 5px;'>{status}</h2>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Gagal memuat atau menjalankan model: {e}")
                st.error("Pastikan Anda sudah menginstall TensorFlow dan file gru_config1.h5 tersedia di folder models.")
