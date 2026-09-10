import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import tensorflow as tf
import sys

# Memastikan modul utils bisa diakses
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.preprocessing import prepare_sequence_from_dict, prepare_sequence_from_df, inverse_scale_temperature

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Jena Climate Command Center - v1.0", layout="wide", page_icon="🌤️")

# Inisialisasi Session State
if 'input_sequence_lstm' not in st.session_state:
    st.session_state['input_sequence_lstm'] = None
if 'input_sequence_gru' not in st.session_state:
    st.session_state['input_sequence_gru'] = None

# --- CACHING MODEL ---
# Workaround untuk error quantization_config pada Dense layer akibat perbedaan versi Keras
class CustomDense(tf.keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        super(CustomDense, self).__init__(**kwargs)

@st.cache_resource
def load_lstm_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm_config1.h5')
    return tf.keras.models.load_model(model_path, custom_objects={'Dense': CustomDense}, compile=False)

@st.cache_resource
def load_gru_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'gru_config1.h5')
    return tf.keras.models.load_model(model_path, custom_objects={'Dense': CustomDense}, compile=False)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("JENA_INTEL // v1.0")
st.sidebar.markdown("**[SYS_ONLINE] | Native Mode**")
st.sidebar.markdown("---")

st.sidebar.markdown("**Active Station Focus:**")
st.sidebar.selectbox("Pilih Stasiun", ["Jena Central [Real]"])

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "01 // OVERVIEW",
    "02 // MODEL_A (LSTM - Prediksi Suhu)",
    "03 // MODEL_B (GRU - Prediksi Suhu)",
    "04 // BENCHMARK"
])

st.sidebar.markdown("---")
st.sidebar.text("Engine: Native Streamlit")
st.sidebar.text("Model Load: Cached")

# --- HALAMAN 01: OVERVIEW ---
if page == "01 // OVERVIEW":
    st.title("🗺️ Geospatial Command Center (Overview)")
    st.write("Versi 1.0 (Baseline): Menampilkan sample dataset dasar Jena Climate.")
    
    st.subheader("Data Observasi Terkini (Preview)")
    @st.cache_data
    def load_data():
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'jena_climate_2009_2016.csv')
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path).tail(100)
        return pd.DataFrame()
        
    df = load_data()
    if not df.empty:
        st.dataframe(df)
        st.line_chart(df[['T (degC)']])
    else:
        st.warning("File data tidak ditemukan di folder `data/`.")

# --- HALAMAN 02: MODEL A (LSTM) ---
elif page == "02 // MODEL_A (LSTM - Prediksi Suhu)":
    st.title("🌡️ Model A Studio (LSTM — Suhu)")
    st.write("Prediksi suhu 6-jam ke depan menggunakan Model LSTM (Sequence 144). Model dilatih khusus untuk data univariat Suhu.")
    
    st.subheader("Data Input")
    input_mode = st.radio("Pilih Mode Input:", ["Manual Input", "CSV Upload"], key="mode_lstm")

    if input_mode == "Manual Input":
        st.write("Masukkan suhu saat ini (akan diduplikasi 144 timesteps):")
        manual_data = {}
        manual_data['T (degC)'] = st.number_input("Suhu T (degC)", value=15.0, key="lstm_t")
        if st.button("Generate Sequence"):
            st.session_state['input_sequence_lstm'] = prepare_sequence_from_dict(manual_data)
            st.success("Sequence berhasil dibentuk dan disimpan.")
            
    elif input_mode == "CSV Upload":
        uploaded_file = st.file_uploader("Unggah dataset cuaca", type=['csv'], key="up_lstm")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if st.button("Generate Sequence dari CSV"):
                try:
                    st.session_state['input_sequence_lstm'] = prepare_sequence_from_df(df)
                    st.success("Sequence berhasil dibentuk dari CSV.")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    st.subheader("Prediction Output")
    if st.button("Jalankan Prediksi LSTM", type="primary"):
        if st.session_state['input_sequence_lstm'] is None:
            st.warning("Generate sequence input terlebih dahulu.")
        else:
            with st.spinner("Menjalankan inferensi LSTM..."):
                try:
                    model = load_lstm_model()
                    
                    start_infer = time.time()
                    prediction = model.predict(st.session_state['input_sequence_lstm'])
                    infer_time = time.time() - start_infer
                    
                    actual_temp = inverse_scale_temperature(float(prediction[0][0]))
                    
                    st.success("Prediksi Berhasil!")
                    st.metric(label="Suhu Terprediksi (T_t+6h)", value=f"{actual_temp:.2f} °C")
                    st.info(f"Inference Latency: {infer_time*1000:.2f} ms")
                except Exception as e:
                    st.error(f"Gagal menjalankan model: {e}")

# --- HALAMAN 03: MODEL B (GRU) ---
elif page == "03 // MODEL_B (GRU - Prediksi Suhu)":
    st.title("🌬️ Model B Studio (GRU — Suhu & Anomali Cuaca)")
    st.write("Prediksi suhu 6-jam ke depan menggunakan Model GRU (Sequence 144). Digunakan sebagai komparasi performa dan detektor anomali.")
    
    st.subheader("Data Input")
    input_mode = st.radio("Pilih Mode Input:", ["Manual Input", "CSV Upload"], key="mode_gru")

    if input_mode == "Manual Input":
        st.write("Masukkan suhu saat ini (akan diduplikasi 144 timesteps):")
        manual_data = {}
        manual_data['T (degC)'] = st.number_input("Suhu T (degC)", value=15.0, key="gru_t")
        if st.button("Generate Sequence"):
            st.session_state['input_sequence_gru'] = prepare_sequence_from_dict(manual_data)
            st.success("Sequence berhasil dibentuk dan disimpan.")
            
    elif input_mode == "CSV Upload":
        uploaded_file = st.file_uploader("Unggah dataset cuaca", type=['csv'], key="up_gru")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if st.button("Generate Sequence dari CSV"):
                try:
                    st.session_state['input_sequence_gru'] = prepare_sequence_from_df(df)
                    st.success("Sequence berhasil dibentuk dari CSV.")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    st.subheader("Prediction Output")
    if st.button("Jalankan Prediksi & Deteksi GRU", type="primary"):
        if st.session_state['input_sequence_gru'] is None:
            st.warning("Generate sequence input terlebih dahulu.")
        else:
            with st.spinner("Menjalankan inferensi GRU..."):
                try:
                    model = load_gru_model()
                    
                    start_infer = time.time()
                    prediction = model.predict(st.session_state['input_sequence_gru'])
                    infer_time = time.time() - start_infer
                    
                    actual_temp = inverse_scale_temperature(float(prediction[0][0]))
                    
                    # Logic Anomali
                    status = "NORMAL (SUHU AMAN)"
                    color = "green"
                    if actual_temp > 30.0:
                        status = "WARNING: PANAS EKSTREM"
                        color = "orange"
                    elif actual_temp < 5.0:
                        status = "CRITICAL: FROST ALERT"
                        color = "red"
                    
                    st.success("Prediksi Berhasil!")
                    st.metric(label="Suhu Terprediksi (T_t+6h)", value=f"{actual_temp:.2f} °C")
                    st.markdown(f"**Indikator Status:** <span style='color: {color}; font-weight:bold;'>{status}</span>", unsafe_allow_html=True)
                    st.info(f"Inference Latency: {infer_time*1000:.2f} ms")
                except Exception as e:
                    st.error(f"Gagal menjalankan model: {e}")

# --- HALAMAN 04: BENCHMARK ---
elif page == "04 // BENCHMARK":
    st.title("⚙️ Benchmark & System Health")
    st.write("Panel optimasi dan metrik performa sistem untuk Baseline v1.0.")
    
    st.subheader("System Footprint (Simulated)")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Model Load Time", value="< 0.1s", delta="Cached", delta_color="normal")
    col2.metric(label="Average Latency", value="~150 ms")
    col3.metric(label="Memory Usage", value="~120 MB")
    
    st.write("Catatan: Di versi v1.0 ini, kita mengimplementasikan `@st.cache_resource` dan state management yang mencegah aplikasi crash saat re-render.")
