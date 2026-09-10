import pandas as pd
import numpy as np

# Z-Score Scaler Constants (Fit on 70% of training data)
TRAIN_MEAN = 9.107596480798954
TRAIN_STD = 8.654227205363727

def scale_temperature(value):
    """Normalisasi nilai suhu menggunakan StandardScaler dari training."""
    return (value - TRAIN_MEAN) / TRAIN_STD

def inverse_scale_temperature(scaled_value):
    """Mengembalikan nilai prediksi ter-skala menjadi Suhu °C aktual."""
    return (scaled_value * TRAIN_STD) + TRAIN_MEAN

def prepare_sequence_from_dict(input_dict, seq_length=144):
    """
    Mengubah single input manual (T (degC)) menjadi sequence array (1, 144, 1).
    """
    raw_temp = float(input_dict.get('T (degC)', 0.0))
    scaled_temp = scale_temperature(raw_temp)
    
    # Buat single step array shape (1,)
    single_step = np.array([scaled_temp])
    
    # Duplikasi menjadi (144, 1)
    sequence = np.tile(single_step, (seq_length, 1))
    
    # Tambahkan dimensi batch (1, 144, 1)
    return np.expand_dims(sequence, axis=0)

def prepare_sequence_from_df(df, seq_length=144):
    """
    Mengambil baris terakhir fitur T (degC) dari dataframe untuk dijadikan sequence prediksi.
    """
    if len(df) < seq_length:
        raise ValueError(f"Dataset minimal harus memiliki {seq_length} baris data.")
        
    if 'T (degC)' not in df.columns:
        raise ValueError("Dataset tidak memiliki kolom 'T (degC)'")
        
    # Ambil fitur T (degC)
    df_temp = df['T (degC)'].tail(seq_length)
    
    # Skalakan data
    scaled_temp = scale_temperature(df_temp.values.astype('float32'))
    
    # Konversi ke shape (144, 1)
    sequence = scaled_temp.reshape(-1, 1)
    
    # Tambahkan dimensi batch (1, 144, 1)
    return np.expand_dims(sequence, axis=0)
