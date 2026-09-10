import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

@st.cache_resource
def init_rag_engine():
    # Menggunakan Groq API untuk latensi ultra-rendah
    llm = ChatGroq(
        temperature=0.2,
        model_name="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY")
    )
    return llm

def generate_climate_insight_stream(predicted_temp, current_features):
    """
    Fungsi generator untuk streaming narasi otomatis hasil prediksi
    """
    llm = init_rag_engine()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Anda adalah pakar meteorologi dan analisis data iklim Jena Climate. 
Tugas Anda adalah memberikan analisis ringkas (maksimal 3 kalimat) dan ramah pengguna awam berdasarkan data sensor dan prediksi model.
Aturan:
1. Langsung ke inti analisis, jangan gunakan salam/pembuka bertele-tele.
2. Jelaskan kaitan antara Suhu (T), Kelembapan (rh), dan Titik Embun (Tdew) jika relevan.
3. Berikan saran praktis jika ada kondisi yang perlu diwaspadai."""),
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
