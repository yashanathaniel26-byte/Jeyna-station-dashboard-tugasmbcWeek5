import tensorflow as tf
import os

print("Memulai konversi model Keras LSTM ke TFLite...")

models_dir = "models"
h5_path = os.path.join(models_dir, "lstm_config1.h5")
tflite_path = os.path.join(models_dir, "lstm_config1_quant.tflite")

if not os.path.exists(h5_path):
    print(f"Error: {h5_path} tidak ditemukan.")
    exit(1)

# Muat model Keras
model = tf.keras.models.load_model(h5_path, compile=False)

# Setup converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16] # Float16 quantization

# Fix for LSTM ops in TFLite
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
converter.experimental_enable_resource_variables = True

# Konversi
tflite_quant_model = converter.convert()

# Simpan model
with open(tflite_path, "wb") as f:
    f.write(tflite_quant_model)

print(f"Berhasil! Model disimpan di {tflite_path}")
