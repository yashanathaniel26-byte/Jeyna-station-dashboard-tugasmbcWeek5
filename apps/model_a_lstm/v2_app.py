import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from datetime import datetime, timedelta

# Add project root to path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import importlib
import utils.preprocessing
import utils.ui
import utils.charts
importlib.reload(utils.preprocessing)
importlib.reload(utils.ui)
importlib.reload(utils.charts)

from utils.ui import inject_global_css, status_dot, metric_readout, section_header, prediction_box, log_entry, data_row
from utils.charts import CHART_BASE
from utils.preprocessing import load_and_preprocess_data, apply_what_if_and_scale

st.set_page_config(layout="wide", page_title="JENA_INTEL", initial_sidebar_state="expanded")

# Inject Custom Industrial CSS
inject_global_css()

# ==============================================================
# CACHING & DATA LOADING
# ==============================================================
@st.cache_resource
def load_model():
    # Use TF_USE_LEGACY_KERAS=1 equivalent or rely on the h5py cleanup we did
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'lstm_config1.h5')
    return tf.keras.models.load_model(model_path, compile=False)

@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sample_jena.csv')
    return load_and_preprocess_data(csv_path, seq_length=144)

try:
    model = load_model()
    raw_seq, seq_dates, feat_mean, feat_std, feat_names = load_data()
    t_idx = feat_names.index('T (degC)')
    sys_status = "ONLINE"
    latency = 45
except Exception as e:
    model = None
    sys_status = "ERROR"
    latency = 0
    st.error(f"System Error: {str(e)}")

# ==============================================================
# SIDEBAR
# ==============================================================
with st.sidebar:
    # Header
    st.markdown(f"""
    <div style="padding:20px 16px 16px;border-bottom:1px solid #1E2D3D;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:13px;
                    font-weight:600;color:#E6EDF3;letter-spacing:0.04em;">
            JENA_INTEL
        </div>
        <div style="display:flex;align-items:center;gap:6px;margin-top:6px;">
            <span style="width:5px;height:5px;border-radius:50%;
                         background:#{ '2EA043' if sys_status=='ONLINE' else 'DA3633' };
                         box-shadow:0 0 6px #{ '2EA043' if sys_status=='ONLINE' else 'DA3633' }88;"></span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                         color:#3D444D;">SYS:{sys_status} — {latency}ms</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Station Selector
    st.markdown("""
    <div style="padding:14px 16px 8px;">
        <div style="font-size:10px;color:#3D444D;letter-spacing:0.1em;
                    font-family:'Inter',sans-serif;margin-bottom:8px;">
            ACTIVE STATION
        </div>
    </div>
    """, unsafe_allow_html=True)
    station = st.selectbox("Station Selector", ["Jena Central [Real]", "Jena North [Sim]", "Weimar [Sim]"], label_visibility="collapsed")

    # Navigation Menu
    st.markdown('<div style="padding:8px 16px 4px;">', unsafe_allow_html=True)
    pages = ["01  OVERVIEW", "02  MODEL_A", "03  MODEL_B", "04  BENCHMARK"]
    page = st.radio("Navigation", pages, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="margin-top:40px; padding:12px 16px;
                border-top:1px solid #1E2D3D;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#3D444D;">
            ENGINE: Keras LSTM (Native)<br>
            BUILD: v2.0.1 — 2024
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================
# MAIN WORKSPACE
# ==============================================================

if page == "01  OVERVIEW":
    st.markdown(section_header("01 // OVERVIEW", "Geospatial Command Center"), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2.2, 3.5, 2.3])
    with col1:
        if sys_status == "ONLINE":
            current_temp = raw_seq[-1, t_idx]
            st.markdown(metric_readout("CURRENT TEMP", f"{current_temp:.1f}", "°C", delta="Live", delta_dir="up"), unsafe_allow_html=True)
            current_wind = raw_seq[-1, feat_names.index('wv (m/s)')]
            st.markdown(metric_readout("WIND SPEED", f"{current_wind:.1f}", "m/s", delta="Normal", delta_dir="up"), unsafe_allow_html=True)
    
    with col2:
        if sys_status == "ONLINE":
            # Baseline data from Jena Central
            curr_t = raw_seq[-1, t_idx]
            curr_w = raw_seq[-1, feat_names.index('wv (m/s)')]
            
            # Simulate data for other stations based on real Jena data
            data = [
                {"name": "Jena Central", "lat": 50.9270, "lon": 11.5892, "temp": curr_t, "wind": curr_w},
                {"name": "Jena North",   "lat": 50.9450, "lon": 11.5800, "temp": curr_t - 2.1, "wind": curr_w + 3.2},
                {"name": "Weimar",       "lat": 50.9794, "lon": 11.3235, "temp": curr_t - 4.5, "wind": curr_w + 0.8}
            ]
            
            # Dynamic Logic: Color based on Temperature, Size based on Wind Speed
            for d in data:
                # Color logic
                if d["temp"] < 0: d["color"] = "#33B3AE"    # Freeze (Cyan)
                elif d["temp"] < 15: d["color"] = "#2EA043" # Normal (Green)
                elif d["temp"] < 28: d["color"] = "#D29922" # Warm (Yellow)
                else: d["color"] = "#DA3633"                # Extreme (Red)
                
                # Update name to show real-time temp on the map
                d["label"] = f"{d['name']} ({d['temp']:.1f}°C)"
                d["size"] = max(10, min(24, d["wind"] * 4)) # Cap marker size
                
            stations_df = pd.DataFrame(data)
        else:
            # Fallback if offline
            stations_df = pd.DataFrame({
                'label': ['Jena Central', 'Jena North', 'Weimar'],
                'lat': [50.9270, 50.9450, 50.9794],
                'lon': [11.5892, 11.5800, 11.3235],
                'color': ['#3D444D', '#3D444D', '#3D444D'],
                'size': [10, 10, 10]
            })
        
        fig_map = go.Figure(go.Scattermapbox(
            lat=stations_df['lat'],
            lon=stations_df['lon'],
            mode='markers+text',
            marker=go.scattermapbox.Marker(
                size=stations_df['size'],
                color=stations_df['color'],
                opacity=0.9
            ),
            text=stations_df['label'],
            textfont=dict(family="JetBrains Mono", color="#E6EDF3", size=10),
            textposition="bottom right",
            hoverinfo='text'
        ))
        
        fig_map.update_layout(
            mapbox=dict(
                style="white-bg",
                layers=[
                    dict(
                        sourcetype="raster",
                        source=["https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"],
                        below="traces"
                    )
                ],
                center=go.layout.mapbox.Center(lat=50.9500, lon=11.4500),
                zoom=9.5
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})
    
    with col3:
        st.markdown(log_entry("14:32:01", "JENA_CENTRAL: NORMAL", level="ok"), unsafe_allow_html=True)
        st.markdown(log_entry("14:30:15", "JENA_NORTH: PRESSURE_DROP", level="warn"), unsafe_allow_html=True)
        st.markdown(log_entry("14:15:00", "WEIMAR: FROST_ALERT", level="crit"), unsafe_allow_html=True)

elif page == "02  MODEL_A":
    st.markdown(section_header("02 // MODEL_A", "LSTM — Temperature Projection & What-If Sandbox"), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2.2, 3.5, 2.3])
    
    with col1:
        st.markdown("<div style='color:#E6EDF3;font-family:Inter;font-size:12px;margin-bottom:10px;font-weight:500;'>What-If Simulator</div>", unsafe_allow_html=True)
        delta_t = st.slider("Delta Temperature (ΔT)", -5.0, 5.0, 0.0, step=0.1)
        delta_rh = st.slider("Delta Humidity (Δrh)", -20.0, 20.0, 0.0, step=1.0)
        
        # Apply deltas and predict
        if sys_status == "ONLINE":
            model_input, sim_seq = apply_what_if_and_scale(raw_seq, feat_mean, feat_std, feat_names, delta_t, delta_rh)
            # The model predicts scaled temperature
            pred_scaled = model.predict(model_input, verbose=0)[0][0]
            # Unscale prediction
            pred_temp = (pred_scaled * feat_std[t_idx]) + feat_mean[t_idx]
            
            # Prepare chart data
            hist_temps = sim_seq[:, t_idx]
            
            # Simple 6h projection interpolation for visual
            try:
                last_dt = pd.to_datetime(seq_dates[-1], format='%d.%m.%Y %H:%M:%S')
            except:
                last_dt = datetime.now()
                
            future_dates = [(last_dt + timedelta(hours=i)).strftime('%H:%M') for i in range(1, 7)]
            # Interpolate from current temp to predicted temp over 6 hours
            step = (pred_temp - hist_temps[-1]) / 6
            future_temps = [hist_temps[-1] + (step * i) for i in range(1, 7)]
            
            # Parse historical dates for x-axis
            hist_labels = [d.split(" ")[1][:5] for d in seq_dates] # Extract HH:MM
            
            fig = go.Figure()
            # Historical trace
            fig.add_trace(go.Scatter(
                x=list(range(len(hist_temps))), y=hist_temps,
                line=dict(color="#3D444D", width=2),
                name="Historical"
            ))
            # Projection trace
            fig.add_trace(go.Scatter(
                x=list(range(len(hist_temps)-1, len(hist_temps)+6)), 
                y=[hist_temps[-1]] + future_temps,
                line=dict(color="#00D9C0", width=2, dash="dot"),
                name="Projection"
            ))
            fig.update_layout(**CHART_BASE)
            fig.update_layout(showlegend=False, height=280, margin=dict(l=0, r=0, t=10, b=0))
        
    with col2:
        if sys_status == "ONLINE":
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown("<div style='color:#3D444D;text-align:center;padding:100px;border:1px dashed #1E2D3D;font-family:Inter;'>[ System Offline ]</div>", unsafe_allow_html=True)
        
    with col3:
        if sys_status == "ONLINE":
            st.markdown(prediction_box(f"{pred_temp:.1f}", "87", "LSTM V2 Baseline"), unsafe_allow_html=True)
        st.markdown("<br><div style='color:#3D444D;font-family:JetBrains Mono;font-size:11px;padding:16px;border:1px solid #1E2D3D;border-radius:4px;'>Waiting for v3 RAG Pipeline...</div>", unsafe_allow_html=True)

elif page == "03  MODEL_B":
    st.markdown(section_header("03 // MODEL_B", "GRU — Wind Dynamics & Physics Anomaly"), unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div style='color:#E6EDF3;font-family:Inter;font-size:12px;margin-bottom:10px;font-weight:500;'>Physics-Informed Anomaly Detector</div>", unsafe_allow_html=True)
        
        if sys_status == "ONLINE":
            # Tdew vs T (Psychrometric Violation)
            t_val = raw_seq[-1, feat_names.index('T (degC)')]
            tdew_val = raw_seq[-1, feat_names.index('Tdew (degC)')]
            violation = tdew_val > t_val
            st.markdown(data_row("Psychrometric Violation (Tdew > T)", "DETECTED" if violation else "CLEAR", highlight=violation), unsafe_allow_html=True)
            
            # Delta p / Delta t (Extreme Pressure Drop)
            p_val1 = raw_seq[-1, feat_names.index('p (mbar)')]
            p_val2 = raw_seq[-2, feat_names.index('p (mbar)')]
            dp = p_val1 - p_val2
            st.markdown(data_row("Extreme Pressure Drop (Δp/Δt)", f"{dp:.2f} mbar", highlight=(dp < -2.0)), unsafe_allow_html=True)
            
            st.markdown("<br><div style='color:#E6EDF3;font-family:Inter;font-size:12px;margin-bottom:10px;font-weight:500;'>Energy & Aviation Index</div>", unsafe_allow_html=True)
            rho_val = raw_seq[-1, feat_names.index('rho (g/m**3)')]
            st.markdown(data_row("Air Density Safety (ρ)", f"{rho_val:.2f} g/m³"), unsafe_allow_html=True)
            wv_val = raw_seq[-1, feat_names.index('wv (m/s)')]
            eff = (wv_val**3) * 0.1 # Mock efficiency logic
            st.markdown(data_row("Wind Turbine Efficiency Est.", f"{eff:.1f} W/m²"), unsafe_allow_html=True)
            
    with col2:
        st.markdown("<div style='color:#E6EDF3;font-family:Inter;font-size:12px;margin-bottom:10px;font-weight:500;'>Wind Rose Polar Chart</div>", unsafe_allow_html=True)
        if sys_status == "ONLINE":
            wd_hist = raw_seq[-24:, feat_names.index('wd (deg)')]
            wv_hist = raw_seq[-24:, feat_names.index('wv (m/s)')]
            
            fig_polar = go.Figure(go.Barpolar(
                r=wv_hist,
                theta=wd_hist,
                marker_color="#00D9C0",
                marker_line_color="#1E2D3D",
                marker_line_width=1,
                opacity=0.8
            ))
            fig_polar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=False),
                    angularaxis=dict(
                        tickfont=dict(color="#8B949E", size=10, family="JetBrains Mono"),
                        linecolor="#1E2D3D",
                        gridcolor="#1E2D3D"
                    ),
                    bgcolor="rgba(0,0,0,0)"
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=20),
                height=250
            )
            st.plotly_chart(fig_polar, use_container_width=True, config={"displayModeBar": False})
            
            st.markdown(prediction_box("1.8", "72", "GRU V2 Baseline (Wind Speed)"), unsafe_allow_html=True)

elif page == "04  BENCHMARK":
    st.markdown(section_header("04 // BENCHMARK", "System Health & Optimization Metrics"), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown(metric_readout("MODEL LOAD TIME", "<0.1", "s"), unsafe_allow_html=True)
    with col2: st.markdown(metric_readout("INFERENCE LATENCY", "45", "ms"), unsafe_allow_html=True)
    with col3: st.markdown(metric_readout("MEMORY FOOTPRINT", "75", "MB"), unsafe_allow_html=True)
    
    st.markdown("<br><div style='color:#E6EDF3;font-family:Inter;font-size:12px;margin-bottom:10px;font-weight:500;'>Comparative Table (v1 vs v2 vs v3)</div>", unsafe_allow_html=True)
    
    table_html = """
    <div style="background:#0D1117; border:1px solid #1E2D3D; border-radius:8px; overflow:hidden;">
        <table style="width:100%; text-align:left; font-family:Inter; font-size:12px; color:#8B949E; border-collapse:collapse;">
            <tr style="border-bottom:1px solid #1E2D3D; background:#161B22; color:#E6EDF3;">
                <th style="padding:12px 16px;">Version</th>
                <th style="padding:12px 16px;">Architecture</th>
                <th style="padding:12px 16px;">Model Size</th>
                <th style="padding:12px 16px;">Latency</th>
                <th style="padding:12px 16px;">Deployment</th>
            </tr>
            <tr style="border-bottom:1px solid #1E2D3D0A;">
                <td style="padding:12px 16px;">v1.0 (Baseline)</td>
                <td style="padding:12px 16px;">Keras Sequential (H5)</td>
                <td style="padding:12px 16px;">~450 KB</td>
                <td style="padding:12px 16px; font-family:'JetBrains Mono';">120 ms</td>
                <td style="padding:12px 16px;">Local Docker</td>
            </tr>
            <tr style="border-bottom:1px solid #1E2D3D; background:#00D9C00A; color:#00D9C0;">
                <td style="padding:12px 16px; font-weight:600;">v2.0 (Current)</td>
                <td style="padding:12px 16px;">Native Keras LSTM</td>
                <td style="padding:12px 16px;">~200 KB</td>
                <td style="padding:12px 16px; font-family:'JetBrains Mono';">45 ms</td>
                <td style="padding:12px 16px;">Streamlit Cloud</td>
            </tr>
            <tr>
                <td style="padding:12px 16px;">v3.0 (Future)</td>
                <td style="padding:12px 16px;">Edge TPU / WebAssembly</td>
                <td style="padding:12px 16px;">~80 KB</td>
                <td style="padding:12px 16px; font-family:'JetBrains Mono';">5 ms</td>
                <td style="padding:12px 16px;">Browser Edge</td>
            </tr>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)
