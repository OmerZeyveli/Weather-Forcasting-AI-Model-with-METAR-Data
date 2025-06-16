from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# === Load Data ===
prefix = sys.argv[1]  # Get prefix from command line
data_path = f"{prefix}input_splits/"

X_train = np.load(f"{data_path}/X_train.npy")
X_val = np.load(f"{data_path}/X_val.npy")
y_train = np.load(f"{data_path}/y_train.npy")
y_val = np.load(f"{data_path}/y_val.npy")


# === Build & Train the model ===
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.2),
    Dense(y_train.shape[1])  # Predict all 9 features
])

model.compile(optimizer="adam", loss='mse', metrics=['mae'])

# Set early stopping
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop]
)


# === Save Model ===
model_output_dir = f"{prefix}model_weights"
os.makedirs(model_output_dir, exist_ok=True)
model.save(f"{model_output_dir}/lstm_weather_model.keras")


# === Evaluate on Validation Set ===
# You can comment this out if you don't want to see validation performance
y_pred = model.predict(X_val)
y_pred = y_pred.reshape((-1, 12, 8))
y_val = y_val.reshape((-1, 12, 8))

step_maes = []
for step in range(12):
    step_mae = mean_absolute_error(y_val[:, step, :], y_pred[:, step, :])
    step_maes.append(step_mae)
    print(f"Step {step + 1}h MAE: {step_mae:.4f}")