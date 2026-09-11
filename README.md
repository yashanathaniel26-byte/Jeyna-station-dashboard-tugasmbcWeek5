# Jena Climate Command Center (V1, V2, V3)

Welcome to the **Jena Climate Command Center** repository. This project is a comprehensive Machine Learning operations (MLOps) dashboard built entirely with **Native Streamlit** and Python. It visualizes climate data from the Jena Weather Station, runs predictive models (LSTM & GRU), and leverages LLM integration (RAG) for real-time natural language analysis.

## 🚀 Features & Versioning

### 🟢 Version 1 (Baseline)
- Data ingestion & preprocessing for Jena Climate dataset.
- Loading of raw `.h5` Keras models for basic inference.
- Default Streamlit vertical layout with basic line charts.

### 🔵 Version 2 (Command Center Redesign)
- **Single-Stack Architecture**: Complete UI overhaul using Streamlit CSS injection without external frontend frameworks like React.
- **Dark Mode & Teal Aesthetics**: Scientific instrument styling (`#00D9C0` accents, `JetBrains Mono` typography).
- **Interactive Geospatial Map**: Plotly Mapbox integration displaying simulated station network conditions.
- **Advanced Visualizations**: Interactive Time Series projections with uncertainty bands, and Wind Rose Polar charts.

### 🟣 Version 3 (Intelligent LLM & Quantized Edge) — *In Progress*
- **Extreme Compression**: Conversion of heavy `.h5` models to quantized `TFLite` formats to drop latency to < 50ms and reduce memory footprint.
- **Ultra-Fast RAG**: Integration with **Groq API** (Llama 3.3) for generating instant, streaming "Executive Briefs". The AI interprets the numerical output from the LSTM/GRU models and produces risk analysis automatically.

## 📂 Repository Structure

```
jena-climate-deployment/
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
├── apps/
│   ├── model_a_lstm/           # Specific entry points (v1, v2) for LSTM
│   ├── model_b_gru/            # Specific entry points (v1, v2) for GRU
│   └── v2_app.py               # Unified V2 Dashboard Entry Point
├── data/
│   ├── sample_jena.csv         # Jena Climate sample dataset
│   └── rag_docs/               # Knowledge base for V3 RAG implementation
├── models/
│   ├── lstm_config1.h5         # Baseline Keras models
│   └── gru_config1.h5
├── utils/
│   ├── charts.py               # Plotly global styling and helpers
│   ├── preprocessing.py        # Array reshaping & scaling helpers
│   ├── rag_engine.py           # (V3) LLM Streaming and RAG implementation
│   └── ui.py                   # Custom Streamlit HTML/CSS components
├── .env                        # API Keys (e.g. GROQ_API_KEY)
├── README.md                   
└── requirements.txt            
```

## ⚙️ Installation & Usage

### 1. Clone & Setup Environment
```bash
git clone https://github.com/username/jena-climate-deployment.git
cd jena-climate-deployment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Dashboard locally
To run the latest unified Version 2 dashboard:
```bash
streamlit run apps/v2_app.py
```

### 3. Setup V3 (LLM Features)
Create a `.env` file in the root directory and add your Groq API key to unlock the Artificial Intelligence features in Version 3:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## ☁️ Deployment

This project is fully optimized for **Streamlit Community Cloud**.
1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Create a new app and point it to the `main` branch of this repository.
3. Set the **Main file path** to `apps/v2_app.py` (or `v3_app.py` once deployed).
4. Add the `GROQ_API_KEY` to the Streamlit Cloud Secrets settings.
