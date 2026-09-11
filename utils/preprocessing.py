import pandas as pd
import numpy as np

def load_and_preprocess_data(csv_path, seq_length=144):
    df = pd.read_csv(csv_path)
    dates = df['Date Time'].values
    features_df = df.drop(columns=['Date Time'])
    feature_names = features_df.columns.tolist()
    features = features_df.values
    
    # Fallback scaler: fitting on the sample data since scaler.pkl is missing
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0 # Prevent division by zero
    
    if len(features) < seq_length:
        raise ValueError(f"Dataset too small. Need {seq_length} rows, got {len(features)}")
        
    raw_sequence = features[-seq_length:].copy()
    sequence_dates = dates[-seq_length:]
    
    return raw_sequence, sequence_dates, mean, std, feature_names

def apply_what_if_and_scale(raw_sequence, mean, std, feature_names, delta_t, delta_rh):
    seq_copy = raw_sequence.copy()
    
    # Find indices
    try:
        t_idx = feature_names.index('T (degC)')
        rh_idx = feature_names.index('rh (%)')
        
        # Apply deltas to the last 24 hours of the sequence for simulation effect
        seq_copy[-24:, t_idx] += delta_t
        seq_copy[-24:, rh_idx] += delta_rh
    except ValueError:
        pass # Columns not found, ignore deltas
        
    # Scale
    scaled_sequence = (seq_copy - mean) / std
    
    # Model only expects Temperature! (Univariate)
    # Shape should be (1, 144, 1)
    model_input = scaled_sequence[:, t_idx:t_idx+1]
    
    return np.expand_dims(model_input, axis=0), seq_copy

# --- LEGACY V1 FUNCTIONS ---
def prepare_sequence_from_dict(manual_data):
    val = manual_data.get('T (degC)', 15.0)
    scaled_val = (val - 9.1) / 8.6
    seq = np.full((1, 144, 1), scaled_val)
    return seq

def prepare_sequence_from_df(df):
    seq = df['T (degC)'].values[-144:]
    scaled_seq = (seq - 9.1) / 8.6
    return np.expand_dims(np.expand_dims(scaled_seq, axis=0), axis=2)

def inverse_scale_temperature(scaled_val):
    return (scaled_val * 8.6) + 9.1
