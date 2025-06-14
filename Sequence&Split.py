import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# === Parameters ===
input_len = 168       # past n hours
output_len = 12       # predict next m hours
test_ratio = 0.2     # 20% for testing
val_ratio = 0.1      # 10% of train set for validation
data_path = "demo/scaled_data/scaled_metar_for_lstm.csv"
output_dir = "demo/output_splits"

# === Load Scaled Data ===
df = pd.read_csv(data_path, parse_dates=["datetime"])
data = df.drop(columns=["datetime"]).values

# === Build Sequences ===
X, y = [], []
for i in range(len(data) - input_len - output_len + 1):
    X.append(data[i:i + input_len])
    y.append(data[i + input_len:i + input_len + output_len])

X = np.array(X)
y = np.array(y)
if output_len > 1:
    y = y.reshape((y.shape[0], y.shape[1] * y.shape[2]))  # (samples, 108)
else:
    y = y.squeeze()

print("Full sequence set:")
print("X:", X.shape)
print("y:", y.shape)

# === Split Train/Test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_ratio, shuffle=False
)

# === Optional: Split Train/Validation ===
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=val_ratio, shuffle=False
)

# === Save Splits ===
os.makedirs(output_dir, exist_ok=True)

np.save(f"{output_dir}/X_train.npy", X_train)
np.save(f"{output_dir}/y_train.npy", y_train)
np.save(f"{output_dir}/X_val.npy", X_val)
np.save(f"{output_dir}/y_val.npy", y_val)
np.save(f"{output_dir}/X_test.npy", X_test)
np.save(f"{output_dir}/y_test.npy", y_test)

# === Summary ===
print("\nFinal split shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)
