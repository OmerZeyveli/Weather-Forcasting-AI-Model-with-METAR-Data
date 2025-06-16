from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import sys

def reconstruct_deg(sin_val, cos_val):
    angle = np.arctan2(sin_val, cos_val) * 180 / np.pi
    return np.mod(angle, 360)

# === Load Data ===
model_weights_folder = sys.argv[1] # Get model weights folder from command line
input_splits_folder = sys.argv[2] # Get input splits folder from command line
metar_processed_folder = sys.argv[3] # Get metar_processed folder for scaler

model = load_model(f"{model_weights_folder}lstm_weather_model.keras")
X_test = np.load(f"{input_splits_folder}X_test.npy")
y_test = np.load(f"{input_splits_folder}y_test.npy")

# === Predict ===
y_pred = model.predict(X_test)

# === Reshape ===
n_samples = y_pred.shape[0]
n_outputs = 12
n_features = 8  # Must match output from model
y_pred = y_pred.reshape((n_samples, n_outputs, n_features))
y_test = y_test.reshape((n_samples, n_outputs, n_features))

# === Load Scaler ===
scaler = joblib.load(f"{metar_processed_folder}scaler.save") # Changed model_weights_folder to metar_processed_folder

# === Fix feature order to match scaler ===
# Scaler was trained on: 
# ["wind_dir_variable", "wind_speed", "visibility", "temperature", "dew_point", "pressure"]
# From y_pred/y_test: indices are
# 0: wind_dir_sin, 1: wind_dir_cos, 2: wind_speed, 3: visibility,
# 4: temperature, 5: dew_point, 6: pressure, 7: wind_dir_variable
indices = [7, 2, 3, 4, 5, 6]  # Rearranged for scaler

# === Subset and inverse transform ===
y_pred_subset = np.stack([y_pred[:, :, i] for i in indices], axis=-1)
y_test_subset = np.stack([y_test[:, :, i] for i in indices], axis=-1)

y_pred_scaled = scaler.inverse_transform(y_pred_subset.reshape(-1, 6)).reshape(n_samples, n_outputs, 6)
y_test_scaled = scaler.inverse_transform(y_test_subset.reshape(-1, 6)).reshape(n_samples, n_outputs, 6)

# === Evaluate Wind Direction ===
y_pred_deg = reconstruct_deg(y_pred[:, :, 0], y_pred[:, :, 1])
y_test_deg = reconstruct_deg(y_test[:, :, 0], y_test[:, :, 1])
wind_dir_mae = mean_absolute_error(y_test_deg, y_pred_deg)

print(f"\nReconstructed Wind Direction MAE (degrees): {wind_dir_mae:.2f}")

# === Evaluate Each Feature ===
feature_names = ["wind_dir_variable", "wind_speed", "visibility", "temperature", "dew_point", "pressure"]
print("\nPer-Feature MAE (original scale):")
for i, name in enumerate(feature_names):
    mae = mean_absolute_error(y_test_scaled[:, :, i], y_pred_scaled[:, :, i])
    print(f"{name:20s} MAE: {mae:.4f}")
    print(f"{name:20s} MAE: {mae:.4f}")
