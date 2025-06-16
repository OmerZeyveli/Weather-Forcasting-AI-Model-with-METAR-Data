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

model = load_model(f"{model_weights_folder}lstm_weather_model.keras")
X_test = np.load(f"{input_splits_folder}X_test.npy")
y_test = np.load(f"{input_splits_folder}y_test.npy")

# === Predict ===
y_pred = model.predict(X_test)

# === Reshape ===
n_samples = y_pred.shape[0]
n_outputs = 12
n_features = 8  # Must match output
y_pred = y_pred.reshape((n_samples, n_outputs, n_features))
y_test = y_test.reshape((n_samples, n_outputs, n_features))

# === Evaluate Wind Direction (not scaled) ===
y_pred_deg = reconstruct_deg(y_pred[:, :, 0], y_pred[:, :, 1])
y_test_deg = reconstruct_deg(y_test[:, :, 0], y_test[:, :, 1])
wind_dir_mae = mean_absolute_error(y_test_deg, y_pred_deg)
print(f"\nReconstructed Wind Direction MAE (degrees): {wind_dir_mae:.2f}")

# === Evaluate MAE on Scaled Features (0-1 range) ===
# Feature order: [wind_dir_sin, wind_dir_cos, wind_speed, visibility, temperature, dew_point, pressure, wind_dir_variable]
scaled_feature_names = [
    "wind_dir_variable", "wind_speed", "visibility", 
    "temperature", "dew_point", "pressure"
]
indices = [7, 2, 3, 4, 5, 6]

print("\nPer-Feature MAE (normalized 0–1):")
for idx, name in zip(indices, scaled_feature_names):
    mae = mean_absolute_error(y_test[:, :, idx], y_pred[:, :, idx])
    print(f"{name:20s} MAE: {mae:.4f}")