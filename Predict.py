import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import sys

# === Reconstruct wind direction from sin/cos
def reconstruct_deg(sin_val, cos_val):
    return np.mod(np.arctan2(sin_val, cos_val) * 180 / np.pi, 360)

# === Arguments ===
model_weights_folder = sys.argv[1]         # e.g., "demo/model_weights/"
input_splits_folder = sys.argv[2]          # e.g., "demo/predict/input_splits/"
metar_processed_folder = sys.argv[3]       # e.g., "demo/predict/metar_processed/"

# === Load model and scaler ===
model = load_model(f"{model_weights_folder}/lstm_weather_model.keras")
scaler = joblib.load(f"{model_weights_folder}/scaler.save")

# === Load most recent test input ===
X_test = np.load(f"{input_splits_folder}/X_test.npy")
X_input = X_test[-1:]  # shape: (1, 168, 12), means only 1 set of 7 day input.

# === Predict ===
y_pred = model.predict(X_input)
y_pred = y_pred.reshape((12, 8))  # 12 hours × 8 features

# === Inverse transform (only real-valued features)
# Indices in model output:
# 0: wind_dir_sin, 1: wind_dir_cos, 2: wind_speed, 3: visibility,
# 4: temperature, 5: dew_point, 6: pressure, 7: wind_dir_variable
real_indices = [7, 2, 3, 4, 5, 6]
subset = np.stack([y_pred[:, i] for i in real_indices], axis=-1)
inv_scaled = scaler.inverse_transform(subset)

# === Reconstruct wind direction
wind_dir_deg = reconstruct_deg(y_pred[:, 0], y_pred[:, 1])

# === Print results
print("\nNext 12 Hours Forecast:")
print("Hour | WindDir(°) | Var | WindSpd | Vis(m) | Temp(°C) | DewPt | Pressure(hPa)")
for i in range(12):
    print(f"{i+1:>4} | {wind_dir_deg[i]:>10.1f} |"
          f" {inv_scaled[i][0]:>3.0f} |"
          f" {inv_scaled[i][1]:>7.2f} |"
          f" {inv_scaled[i][2]:>7.0f} |"
          f" {inv_scaled[i][3]:>7.2f} |"
          f" {inv_scaled[i][4]:>6.2f} |"
          f" {inv_scaled[i][5]:>12.2f}")
