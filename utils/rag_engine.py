import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

@st.cache_resource
def init_rag_engine():
    """Menginisialisasi engine LLM OpenAI"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY tidak ditemukan. Harap atur di file .env atau Streamlit Secrets.")
        
    llm = ChatOpenAI(
        temperature=0.2,
        model_name="gpt-4o-mini",
        api_key=api_key
    )
    return llm

def generate_climate_insight_stream(predicted_temp, current_features):
    """
    Fungsi generator untuk streaming narasi otomatis hasil prediksi
    """
    llm = init_rag_engine()
    
    # Baca file Knowledge Base
    kb_path = os.path.join(os.path.dirname(__file__), "..", "data", "rag_docs", "jena_climate_knowledge.txt")
    knowledge_base = ""
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            knowledge_base = f.read()
            
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Anda adalah pakar meteorologi dan analisis data iklim Jena Climate. 
Tugas Anda adalah memberikan analisis ringkas (maksimal 3 kalimat) dan ramah pengguna awam berdasarkan data sensor dan prediksi model.
Gunakan Panduan Fisika Cuaca berikut sebagai referensi dasar Anda:

=== JENA CLIMATE KNOWLEDGE BASE ===
{knowledge_base}
===================================

Aturan:
1. Langsung ke inti analisis, jangan gunakan salam/pembuka bertele-tele.
2. Jelaskan kaitan antara Suhu (T), Kelembapan (rh), dan Titik Embun (Tdew) jika relevan berdasarkan pedoman di atas.
3. Berikan rekomendasi operasional atau peringatan jika kondisi fisika cuaca mengindikasikan risiko."""),
        ("user", """
Data Sensor Jena Saat Ini:
- Tekanan Udara (p): {p} mbar
- Suhu Saat Ini (T): {T} °C
- Titik Embun (Tdew): {Tdew} °C
- Kelembapan Relatif (rh): {rh} %
- Kecepatan Angin (wv): {wv} m/s

Hasil Prediksi Model LSTM/GRU (6 Jam Ke Depan):
- Prediksi Suhu: {pred_temp} °C

Berikan analisis dan rekomendasi ringkas!""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    # Return generator stream untuk st.write_stream
    return chain.stream({
        "p": current_features.get("p (mbar)", 980),
        "T": current_features.get("T (degC)", 15.0),
        "Tdew": current_features.get("Tdew (degC)", 10.0),
        "rh": current_features.get("rh (%)", 70.0),
        "wv": current_features.get("wv (m/s)", 2.1),
        "pred_temp": predicted_temp
    })
