import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

# === Parameters ===
input_len = 168  # past 168 hours (7 days)
output_len = 12  # predict next 12 hours

# === Command-line arguments ===
processed_metar_folder = sys.argv[1]
input_splits_folder = sys.argv[2]
test_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
val_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1

data_path = f"{processed_metar_folder}scaled_metar_for_lstm.csv"
df = pd.read_csv(data_path, parse_dates=["datetime"])
os.makedirs(input_splits_folder, exist_ok=True)

# === Features ===
input_features = [
    "wind_dir_variable", "wind_speed", "visibility",
    "temperature", "dew_point", "pressure",
    "wind_dir_sin", "wind_dir_cos",
    "hour_sin", "hour_cos", "month_sin", "month_cos"
]

target_features = [
    "wind_dir_sin", "wind_dir_cos",
    "wind_speed", "visibility", "temperature",
    "dew_point", "pressure", "wind_dir_variable"
]

X_data = df[input_features].values
y_data = df[target_features].values

# === Build Sequences ===
X, y = [], []
for i in range(len(df) - input_len - output_len + 1):
    X.append(X_data[i:i + input_len])
    y.append(y_data[i + input_len:i + input_len + output_len])

X = np.array(X)
y = np.array(y).reshape((-1, output_len * len(target_features)))

print("Full sequence set:")
print("X:", X.shape)
print("y:", y.shape)

# === If full test set only (test_ratio=1.0 and val_ratio=0), skip splitting ===
if test_ratio == 1.0 and val_ratio == 0.0:
    np.save(f"{input_splits_folder}/X_test.npy", X)
    np.save(f"{input_splits_folder}/y_test.npy", y)
    print("\nSaved entire dataset as test set.")
    print("X_test:", X.shape)
    print("y_test:", y.shape)

else:
    # === Split Train/Test ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=False
    )

    # === Split Train/Validation ===
    if val_ratio > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_ratio, shuffle=False
        )

        # Save val
        np.save(f"{input_splits_folder}/X_val.npy", X_val)
        np.save(f"{input_splits_folder}/y_val.npy", y_val)

    # Save all other splits
    np.save(f"{input_splits_folder}/X_train.npy", X_train)
    np.save(f"{input_splits_folder}/y_train.npy", y_train)
    np.save(f"{input_splits_folder}/X_test.npy", X_test)
    np.save(f"{input_splits_folder}/y_test.npy", y_test)

    print("\nFinal split shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    if val_ratio > 0:
        print("X_val  :", X_val.shape)
        print("y_val  :", y_val.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)
