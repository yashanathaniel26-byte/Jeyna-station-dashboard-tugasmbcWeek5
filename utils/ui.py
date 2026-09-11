import streamlit as st

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* Reset Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { padding: 0; }

/* Base typography */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #1E2D3D !important;
    min-width: 220px !important;
    max-width: 220px !important;
}

/* Input fields */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-baseweb="select"] {
    background: #161B22 !important;
    border: 1px solid #1E2D3D !important;
    border-radius: 4px !important;
    color: #E6EDF3 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: #2A4460 !important;
    box-shadow: 0 0 0 1px #00D9C030 !important;
}

/* Slider */
[data-baseweb="slider"] [role="slider"] {
    background: #00D9C0 !important;
    border-color: #00D9C0 !important;
}
[data-baseweb="slider"] div[data-testid="stSlider"] div {
    background: #1E2D3D !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid #1E2D3D !important;
    border-radius: 4px !important;
    color: #8B949E !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    transition: border-color 0.15s, color 0.15s !important;
}
.stButton > button:hover {
    border-color: #00D9C0 !important;
    color: #00D9C0 !important;
    background: #00D9C008 !important;
}

/* Primary button variant */
.btn-primary .stButton > button {
    background: #00D9C015 !important;
    border-color: #00D9C040 !important;
    color: #00D9C0 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1E2D3D !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #3D444D !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 6px 14px !important;
    font-family: 'Inter', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    color: #00D9C0 !important;
    border-bottom: 2px solid #00D9C0 !important;
    background: transparent !important;
}

/* Selectbox */
[data-baseweb="select"] > div {
    background: #161B22 !important;
    border: 1px solid #1E2D3D !important;
    border-radius: 4px !important;
}

/* Metric (fallback) */
[data-testid="metric-container"] {
    background: #0D1117 !important;
    border: 1px solid #1E2D3D !important;
    border-radius: 8px !important;
    padding: 14px 16px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 20px !important;
    color: #E6EDF3 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 10px !important;
    color: #3D444D !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Divider */
hr { border-color: #1E2D3D !important; margin: 12px 0 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #080B0F; }
::-webkit-scrollbar-thumb { background: #1E2D3D; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #2A4460; }

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #1E2D3D !important;
    border-radius: 4px !important;
    background: #0D1117 !important;
}
[data-testid="stExpander"] summary {
    font-size: 12px !important;
    color: #8B949E !important;
    font-family: 'Inter', sans-serif !important;
}
</style>
"""

def inject_global_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

def status_dot(label, state="ok"):
    colors = {
        "ok":   "#2EA043",
        "warn": "#D29922",
        "crit": "#DA3633",
        "off":  "#3D444D",
    }
    color = colors.get(state, colors["off"])
    return f"""
    <div style="display:flex;align-items:center;gap:6px;">
        <span style="width:6px;height:6px;border-radius:50%;
                     background:{color};
                     box-shadow:0 0 6px {color}88;
                     flex-shrink:0;"></span>
        <span style="font-family:'Inter',sans-serif;font-size:11px;
                     color:#8B949E;">{label}</span>
    </div>
    """

def metric_readout(label, value, unit="", delta=None, delta_dir="up"):
    delta_color = "#2EA043" if delta_dir == "up" else "#DA3633"
    delta_html = f'<span style="font-size:11px;color:{delta_color};font-family:Inter;">{delta}</span>' if delta else ""
    return f"""
    <div style="padding:14px 16px;background:#0D1117;
                border:1px solid #1E2D3D;border-radius:8px;">
        <div style="font-family:'Inter',sans-serif;font-size:10px;
                    color:#3D444D;letter-spacing:0.08em;
                    text-transform:uppercase;margin-bottom:6px;">{label}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:22px;
                    font-weight:600;color:#E6EDF3;line-height:1;">
            {value}<span style="font-size:12px;color:#8B949E;margin-left:4px;">{unit}</span>
        </div>
        <div style="margin-top:4px;">{delta_html}</div>
    </div>
    """

def section_header(title, subtitle=""):
    sub_html = f'<div style="font-size:11px;color:#8B949E;margin-top:2px;">{subtitle}</div>' if subtitle else ""
    return f"""
    <div style="padding-bottom:10px;border-bottom:1px solid #1E2D3D;margin-bottom:14px;">
        <div style="font-family:'Inter',sans-serif;font-size:11px;
                    font-weight:600;color:#E6EDF3;letter-spacing:0.02em;">{title}</div>
        {sub_html}
    </div>
    """

def log_entry(timestamp, message, level="info"):
    colors = {"info": "#8B949E", "warn": "#D29922", "crit": "#DA3633", "ok": "#2EA043"}
    prefix = {"info": "INFO", "warn": "WARN", "crit": "CRIT", "ok": "OK  "}
    color = colors.get(level, colors["info"])
    pre   = prefix.get(level, "INFO")
    return f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                color:#8B949E;padding:4px 0;border-bottom:1px solid #1E2D3D0A;">
        <span style="color:#3D444D;">{timestamp}</span>
        <span style="color:{color};margin:0 8px;">[{pre}]</span>
        <span>{message}</span>
    </div>
    """

def prediction_box(temp_pred, confidence, model_name):
    return f"""
    <div style="background:#00D9C015;border:1px solid #00D9C030;
                border-radius:8px;padding:16px;">
        <div style="font-size:10px;color:#00D9C0;font-family:'Inter',sans-serif;
                    letter-spacing:0.08em;text-transform:uppercase;margin-bottom:10px;">
            {model_name} — Projection t+6h
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:32px;
                    font-weight:600;color:#E6EDF3;line-height:1;">
            {temp_pred}<span style="font-size:16px;color:#8B949E;">°C</span>
        </div>
        <div style="margin-top:8px;font-size:11px;color:#8B949E;font-family:'Inter',sans-serif;">
            Confidence: <span style="color:#00D9C0;font-family:'JetBrains Mono',monospace;">{confidence}%</span>
        </div>
    </div>
    """

def data_row(label, value, highlight=False):
    bg = "#161B22" if highlight else "transparent"
    return f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:7px 10px;background:{bg};border-radius:4px;">
        <span style="font-family:'Inter',sans-serif;font-size:12px;color:#8B949E;">{label}</span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#E6EDF3;">{value}</span>
    </div>
    """
