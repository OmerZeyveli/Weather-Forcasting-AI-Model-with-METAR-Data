from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
import numpy as np
import joblib

# === Load model and data ===
model = load_model("demo/trained_model/lstm_weather_model.keras")
X_test = np.load("demo/output_splits/X_test.npy")
y_test = np.load("demo/output_splits/y_test.npy")

# === Predict ===
y_pred = model.predict(X_test)

# === Reshape ===
n_samples = y_pred.shape[0]
n_outputs = 12  # steps
n_features = 9  # features per step

y_pred = y_pred.reshape((n_samples, n_outputs, n_features))
y_test = y_test.reshape((n_samples, n_outputs, n_features))

# === Load scaler and inverse transform ===
scaler = joblib.load("demo/scaled_data/scaler.save")
y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, n_features)).reshape(n_samples, n_outputs, n_features)
y_test_orig = scaler.inverse_transform(y_test.reshape(-1, n_features)).reshape(n_samples, n_outputs, n_features)

# === Feature names (adjust order if needed) ===
feature_names = [
    "wind_dir", "wind_dir_variable", "wind_speed", "visibility",
    "temperature", "dew_point", "pressure", "hour", "month"
]

print("Per-Feature MAE (original scale):")
for i, name in enumerate(feature_names):
    mae = mean_absolute_error(y_test_orig[:, :, i], y_pred_orig[:, :, i])
    print(f"{name:20s} MAE: {mae:.4f}")