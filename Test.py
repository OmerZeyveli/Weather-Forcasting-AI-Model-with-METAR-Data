from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
import numpy as np
import joblib

def reconstruct_deg(sin_val, cos_val):
    angle = np.arctan2(sin_val, cos_val) * 180 / np.pi
    return np.mod(angle, 360)

model = load_model("demo/model_weights/lstm_weather_model.keras")
X_test = np.load("demo/input_splits/X_test.npy")
y_test = np.load("demo/input_splits/y_test.npy")

y_pred = model.predict(X_test)

n_samples = y_pred.shape[0]
n_outputs = 12
n_features = 8

y_pred = y_pred.reshape((n_samples, n_outputs, n_features))
y_test = y_test.reshape((n_samples, n_outputs, n_features))

scaler = joblib.load("demo/model_weights/scaler.save")

# Inverse transform only the scaled part (first 6 features)
y_pred_scaled = scaler.inverse_transform(y_pred[:, :, 2:8].reshape(-1, 6)).reshape(n_samples, n_outputs, 6)
y_test_scaled = scaler.inverse_transform(y_test[:, :, 2:8].reshape(-1, 6)).reshape(n_samples, n_outputs, 6)

# Evaluate wind_dir in degrees from sin/cos
y_pred_deg = reconstruct_deg(y_pred[:, :, 0], y_pred[:, :, 1])
y_test_deg = reconstruct_deg(y_test[:, :, 0], y_test[:, :, 1])
wind_dir_mae = mean_absolute_error(y_test_deg, y_pred_deg)

print(f"Reconstructed Wind Direction MAE (degrees): {wind_dir_mae:.2f}")

feature_names = ["wind_speed", "visibility", "temperature", "dew_point", "pressure", "wind_dir_variable"]
print("Per-Feature MAE (original scale):")
for i, name in enumerate(feature_names):
    mae = mean_absolute_error(y_test_scaled[:, :, i], y_pred_scaled[:, :, i])
    print(f"{name:20s} MAE: {mae:.4f}")
