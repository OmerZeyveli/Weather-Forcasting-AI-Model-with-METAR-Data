from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the splits (if in separate script)
X_train = np.load("demo/output_splits/X_train.npy")
X_val = np.load("demo/output_splits/X_val.npy")
y_train = np.load("demo/output_splits/y_train.npy")
y_val = np.load("demo/output_splits/y_val.npy")

# Build the model
# model = Sequential([
#     LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
#     Dropout(0.2),
#     Dense(y_train.shape[1])  # Predict all 9 features
# ])

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
os.makedirs("demo/trained_model", exist_ok=True)
model.save("demo/trained_model/lstm_weather_model.keras")

# # === Plot training loss ===
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
# plt.title("Loss Curve")
# plt.xlabel("Epoch")
# plt.ylabel("Loss (MSE)")
# plt.legend()
# plt.grid(True)
# plt.show()

# === Predict and reshape for evaluation ===
y_pred = model.predict(X_val)

# Reshape both to (samples, 12, 9)
y_pred = y_pred.reshape((-1, 12, 9))
y_val = y_val.reshape((-1, 12, 9))

# === Evaluate MAE per hour ===
step_maes = []
for step in range(12):
    step_mae = mean_absolute_error(y_val[:, step, :], y_pred[:, step, :])
    step_maes.append(step_mae)
    print(f"Step {step + 1}h MAE: {step_mae:.4f}")

# # === Plot MAE per forecast hour ===
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, 13), step_maes, marker='o')
# plt.title("MAE per Forecast Hour")
# plt.xlabel("Forecast Hour")
# plt.ylabel("Mean Absolute Error")
# plt.grid(True)
# plt.xticks(range(1, 13))
# plt.show()