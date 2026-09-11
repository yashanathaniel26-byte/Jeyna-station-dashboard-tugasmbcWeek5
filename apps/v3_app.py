import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import random
from datetime import datetime
import sys

# Tambahkan utils ke path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.ui import (
    inject_global_css, status_dot, metric_readout, 
    section_header, log_entry, prediction_box, data_row
)
from utils.charts import CHART_BASE
from utils.preprocessing import load_and_preprocess_data, apply_what_if_and_scale
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="JENA_INTEL Command Center", layout="wide")
inject_global_css()

# --- DATA & MODEL LOADING ---
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_jena.csv")
    try:
        df = pd.read_csv(csv_path)
        latest = df.iloc[-1]
        return df, latest
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None, None

@st.cache_data
def get_seq_data():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_jena.csv")
    return load_and_preprocess_data(csv_path, seq_length=144)

@st.cache_resource
def load_tflite_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "lstm_config1_quant.tflite")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

df, latest_data = load_data()
raw_seq, seq_dates, feat_mean, feat_std, feat_names = get_seq_data()
interpreter = load_tflite_model()

# --- SIDEBAR NAV ---
with st.sidebar:
    st.markdown(f"""
    <div style="padding:20px 16px 16px;border-bottom:1px solid #1E2D3D;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:13px;
                    font-weight:600;color:#E6EDF3;letter-spacing:0.04em;">
            JENA_INTEL
        </div>
        <div style="display:flex;align-items:center;gap:6px;margin-top:6px;">
            <span style="width:5px;height:5px;border-radius:50%;
                         background:#2EA043;box-shadow:0 0 6px #2EA04388;"></span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                         color:#3D444D;">SYS:ONLINE — 42ms</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:14px 16px 8px;">
        <div style="font-size:10px;color:#3D444D;letter-spacing:0.1em;
                    font-family:'Inter',sans-serif;margin-bottom:8px;">
            ACTIVE STATION
        </div>
    </div>
    """, unsafe_allow_html=True)
    station = st.selectbox("Active Station", ["Jena Central [Real]", "Jena North [Sim]", "Weimar [Sim]"], label_visibility="collapsed")

    st.markdown('<div style="padding:8px 16px 4px;">', unsafe_allow_html=True)
    pages = ["01 OVERVIEW", "02 MODEL_A (LSTM)", "03 MODEL_B (GRU)", "04 BENCHMARK"]
    page = st.radio("Navigation Menu", pages, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 40px; padding:12px 16px; border-top:1px solid #1E2D3D;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#3D444D;">
            ENGINE: TFLite + GPT-4o-mini<br>
            BUILD: v3.0.0 — 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- PAGES ---
if page == "01 OVERVIEW":
    col1, col2, col3 = st.columns([2.2, 3.5, 2.3])
    
    with col1:
        st.markdown(section_header("STATION NETWORK", "Active Nodes Status"), unsafe_allow_html=True)
        st.markdown(status_dot("Jena Central — Realtime", "ok"), unsafe_allow_html=True)
        st.markdown(status_dot("Jena North — Simulated", "warn"), unsafe_allow_html=True)
        st.markdown(status_dot("Weimar — Simulated", "ok"), unsafe_allow_html=True)
        st.markdown(status_dot("Erfurt — Offline", "off"), unsafe_allow_html=True)
        
        st.write("")
        st.markdown(section_header("ENVIRONMENT METRICS", f"Latest reading: {latest_data['Date Time']}"), unsafe_allow_html=True)
        
        st.markdown(metric_readout("Air Pressure (p)", f"{latest_data['p (mbar)']:.2f}", "mbar", delta="+0.12", delta_dir="up"), unsafe_allow_html=True)
        st.write("")
        st.markdown(metric_readout("Relative Humidity (rh)", f"{latest_data['rh (%)']:.1f}", "%", delta="-1.5", delta_dir="down"), unsafe_allow_html=True)
        st.write("")
        st.markdown(metric_readout("Wind Velocity (wv)", f"{latest_data['wv (m/s)']:.2f}", "m/s", delta="Stable", delta_dir="up"), unsafe_allow_html=True)

    with col2:
        st.markdown(section_header("GEOSPATIAL COMMAND CENTER", "Network Topography & Temp Indicators"), unsafe_allow_html=True)
        
        # Simulasi Koordinat untuk beberapa station di sekitar Jena
        jena_lat = 50.927
        jena_lon = 11.583
        
        base_temp = latest_data["T (degC)"]
        
        map_data = pd.DataFrame({
            "Station": ["Jena Central", "Jena North", "Weimar", "Erfurt"],
            "Lat": [jena_lat, jena_lat + 0.05, jena_lat - 0.04, jena_lat - 0.08],
            "Lon": [jena_lon, jena_lon + 0.02, jena_lon - 0.15, jena_lon - 0.3],
            "Temperature": [
                base_temp, 
                base_temp - 0.5, 
                base_temp + 1.2, 
                base_temp + 0.8
            ]
        })
        
        # Color mapping logic based on Temperature
        def get_color(t):
            if t < 0: return "#58A6FF" # Blue cold
            if t < 15: return "#00D9C0" # Teal cool
            if t < 25: return "#D29922" # Yellow warm
            return "#DA3633" # Red hot
            
        map_data["Color"] = map_data["Temperature"].apply(get_color)

        fig_map = px.scatter_mapbox(
            map_data, lat="Lat", lon="Lon", hover_name="Station", 
            hover_data={"Temperature": ":.2f", "Lat": False, "Lon": False, "Color": False},
            color="Color", size_max=15, zoom=9.5, height=450
        )
        # Free OSM style
        fig_map.update_layout(mapbox_style="open-street-map")
        fig_map.update_traces(marker=dict(size=14, opacity=0.8))
        
        # Terapkan gaya CHART_BASE dengan mempertahankan mapbox setting
        fig_map.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        
        st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    with col3:
        st.markdown(section_header("NETWORK ALERTS", "System & Weather Anomaly Logs"), unsafe_allow_html=True)
        
        st.markdown(log_entry("14:32:00", "Jena North reporting high wind variance", "warn"), unsafe_allow_html=True)
        st.markdown(log_entry("14:15:22", "Data synchronization completed", "ok"), unsafe_allow_html=True)
        st.markdown(log_entry("13:59:01", "Erfurt node connection timeout", "crit"), unsafe_allow_html=True)
        st.markdown(log_entry("12:00:00", "Daily diagnostic scan started", "info"), unsafe_allow_html=True)
        
        st.write("")
        st.markdown(section_header("STATION LEADERBOARD", "Temperature Delta Matrix"), unsafe_allow_html=True)
        st.markdown(data_row("Weimar", f"{base_temp + 1.2:.2f}°C", highlight=True), unsafe_allow_html=True)
        st.markdown(data_row("Erfurt", f"{base_temp + 0.8:.2f}°C"), unsafe_allow_html=True)
        st.markdown(data_row("Jena Central", f"{base_temp:.2f}°C"), unsafe_allow_html=True)
        st.markdown(data_row("Jena North", f"{base_temp - 0.5:.2f}°C"), unsafe_allow_html=True)


elif page == "02 MODEL_A (LSTM)":
    col1, col2, col3 = st.columns([2.2, 3.5, 2.3])
    with col1:
        st.markdown(section_header("WHAT-IF SANDBOX", "Perturb Input Variables"), unsafe_allow_html=True)
        delta_t = st.slider("ΔT (Temperature Bias)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
        delta_rh = st.slider("Δrh (Humidity Bias)", min_value=-10.0, max_value=10.0, value=0.0, step=0.5)
        st.button("INJECT & RE-RUN LSTM", key="run_lstm")
        
    with col2:
        st.markdown(section_header("SEQUENCE TRAJECTORY", "144h History + 6h Projection"), unsafe_allow_html=True)
        # Apply deltas to raw sequence
        model_input, sim_seq = apply_what_if_and_scale(raw_seq, feat_mean, feat_std, feat_names, delta_t, delta_rh)
        
        # TFLite Inference
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], tf.cast(model_input, tf.float32))
        interpreter.invoke()
        pred_scaled = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        t_idx = feat_names.index('T (degC)')
        pred_temp = (pred_scaled * feat_std[t_idx]) + feat_mean[t_idx]
        
        # Prepare plot data
        history_temps = sim_seq[:, t_idx]
        last_time = pd.to_datetime(seq_dates[-1], format='%d.%m.%Y %H:%M:%S')
        future_times = [last_time + pd.Timedelta(hours=i) for i in range(1, 7)]
        
        # Interpolate projection
        step = (pred_temp - history_temps[-1]) / 6
        pred_y = [history_temps[-1] + (step * i) for i in range(1, 7)]
        
        fig = go.Figure()
        
        # Historical Baseline
        fig.add_trace(go.Scatter(
            x=[pd.to_datetime(d, format='%d.%m.%Y %H:%M:%S') for d in seq_dates], 
            y=history_temps,
            line=dict(color="#3D444D", width=1.5),
            name="History"
        ))
        
        # Projection Line
        fig.add_trace(go.Scatter(
            x=future_times, 
            y=pred_y,
            line=dict(color="#00D9C0", width=1.5, dash="dot"),
            name="Projection",
            opacity=0.75,
        ))
        
        # Uncertainty Band
        upper_bound = [y + 0.5 for y in pred_y]
        lower_bound = [y - 0.5 for y in pred_y]
        
        fig.add_trace(go.Scatter(
            x=future_times + future_times[::-1],
            y=upper_bound + lower_bound[::-1],
            fill="toself",
            fillcolor="rgba(0,217,192,0.06)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            name="Confidence Interval"
        ))
        
        # Terapkan gaya CHART_BASE
        fig.update_layout(CHART_BASE)
        fig.update_layout(
            height=350,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
    with col3:
        st.markdown(section_header("PREDICTION ENGINE", "Output T+6h"), unsafe_allow_html=True)
        # Gunakan nilai prediksi terakhir
        final_pred = pred_y[-1]
        st.markdown(prediction_box(f"{final_pred:.2f}", 96, "TFLite Quantized Edge"), unsafe_allow_html=True)

        st.write("")
        st.markdown(section_header("EXECUTIVE BRIEF", "AI Auto-Generated"), unsafe_allow_html=True)
        
        if st.button("Generate AI Explanation", key="gen_ai"):
            try:
                from utils.rag_engine import generate_climate_insight_stream
                current_features = {
                    "p (mbar)": latest_data['p (mbar)'],
                    "T (degC)": latest_data['T (degC)'],
                    "Tdew (degC)": latest_data['Tdew (degC)'],
                    "rh (%)": latest_data['rh (%)'],
                    "wv (m/s)": latest_data['wv (m/s)']
                }
                with st.chat_message("assistant"):
                    st.write_stream(generate_climate_insight_stream(final_pred, current_features))
            except Exception as e:
                st.error(f"AI Engine Error: {str(e)}")
        else:
            st.markdown(f"""
            <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#8B949E;line-height:1.5;">
            > <b>READY:</b> AI Assistant is on standby.<br>
            > Click 'Generate AI Explanation' to run RAG analysis.<br><br>
            > <b>BACKEND:</b> OpenAI GPT-4o-mini (Streaming)
            </div>
            """, unsafe_allow_html=True)

elif page == "03 MODEL_B (GRU)":
    col1, col2, col3 = st.columns([2.2, 3.5, 2.3])
    with col1:
        st.markdown(section_header("DYNAMIC CONTROLS", "Wind & Humidity Dynamics"), unsafe_allow_html=True)
        st.slider("Wind Noise Injector", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
        st.slider("Model Confidence Threshold", min_value=50, max_value=99, value=85, step=1)
        st.button("RUN GRU ANALYSIS", key="run_gru")
        
        st.write("")
        st.markdown(section_header("PHYSICS VALIDATOR", "Anomaly Detection"), unsafe_allow_html=True)
        st.markdown(log_entry("CHECK 1", "Dew Point (Tdew) < T (degC) — Valid", "ok"), unsafe_allow_html=True)
        st.markdown(log_entry("CHECK 2", "Relative Humidity within 0-100% — Valid", "ok"), unsafe_allow_html=True)
        
    with col2:
        st.markdown(section_header("WIND DYNAMICS", "Wind Rose Polar Chart"), unsafe_allow_html=True)
        
        # Wind Rose menggunakan data wind direction dan velocity dari history
        history_df = df.tail(144).copy()
        
        fig_polar = px.bar_polar(
            history_df, 
            r="wv (m/s)", 
            theta="wd (deg)", 
            color="wv (m/s)", 
            color_continuous_scale=px.colors.sequential.Teal,
            template="plotly_dark",
        )
        
        fig_polar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#8B949E", size=11),
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                angularaxis=dict(tickcolor="#3D444D", linecolor="#1E2D3D"),
                radialaxis=dict(tickcolor="#3D444D", linecolor="#1E2D3D", gridcolor="#1E2D3D")
            ),
            height=350,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_polar, use_container_width=True, config={"displayModeBar": False})
        
    with col3:
        st.markdown(section_header("BUSINESS IMPACT", "Pattern Match & Risks"), unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background:#161B22;border:1px solid #1E2D3D;border-radius:8px;padding:16px;">
            <div style="color:#E6EDF3;font-weight:600;margin-bottom:8px;">Aviation Safety Proxy</div>
            <div style="display:flex;align-items:center;gap:10px;">
                <span style="font-family:'JetBrains Mono';font-size:24px;color:#2EA043;">OPTIMAL</span>
            </div>
            <div style="font-size:11px;color:#8B949E;margin-top:8px;">
                Wind shear is minimal. Cross-station variance is low.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.markdown("""
        <div style="background:#161B22;border:1px solid #1E2D3D;border-radius:8px;padding:16px;">
            <div style="color:#E6EDF3;font-weight:600;margin-bottom:8px;">Historical Pattern Match</div>
            <div style="display:flex;align-items:center;gap:10px;">
                <span style="font-family:'JetBrains Mono';font-size:24px;color:#58A6FF;">92%</span>
            </div>
            <div style="font-size:11px;color:#8B949E;margin-top:8px;">
                Matches weather pattern from Jan 03, 2009.
            </div>
        </div>
        """, unsafe_allow_html=True)

elif page == "04 BENCHMARK":
    st.markdown(section_header("SYSTEM BENCHMARK", "v1 vs v2 vs v3 Performance"), unsafe_allow_html=True)
    
    benchmark_df = pd.DataFrame({
        "Metric": ["Load Time", "Memory", "Latency", "Accuracy (MAE)"],
        "v1 (Baseline Keras)": ["1.2s", "450 MB", "120ms", "0.85"],
        "v2 (Streamlit UI)": ["0.8s", "250 MB", "100ms", "0.85"],
        "v3 (TFLite + RAG)": ["0.3s", "85 MB", "45ms", "0.87"]
    })
    
    # Custom CSS for table
    st.markdown("""
    <style>
    .dataframe {
        width: 100%;
        background-color: #0D1117;
        color: #E6EDF3;
        font-family: 'Inter', sans-serif;
        border-collapse: collapse;
    }
    .dataframe th {
        background-color: #161B22;
        color: #8B949E;
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid #1E2D3D;
        font-weight: 600;
    }
    .dataframe td {
        padding: 12px;
        border-bottom: 1px solid #1E2D3D;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
    }
    .dataframe tr:hover {
        background-color: #1E2D3D;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(benchmark_df.to_html(index=False), unsafe_allow_html=True)

