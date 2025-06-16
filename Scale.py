# Scaler.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import os
import sys

# === Load your processed METAR file ===
processed_metar_folder = sys.argv[1] # Get processed metar folder from command line

df = pd.read_csv(f"{processed_metar_folder}processed_metar_for_lstm.csv", parse_dates=["datetime"])

# === Sort by datetime (just in case) ===
df = df.sort_values("datetime").reset_index(drop=True)

# === Interpolate missing temperature and dew_point ===
df["temperature"] = df["temperature"].interpolate(method="linear")
df["dew_point"] = df["dew_point"].interpolate(method="linear")
df["wind_dir"] = df["wind_dir"].interpolate(method="linear")

# === Cyclical encoding ===
df["wind_dir_sin"] = np.sin(2 * np.pi * df["wind_dir"] / 360)
df["wind_dir_cos"] = np.cos(2 * np.pi * df["wind_dir"] / 360)

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# === Drop original cyclical columns ===
df = df.drop(columns=["wind_dir", "hour", "month"])

# === Select features to scale ===
features = [
    "wind_dir_variable", "wind_speed", "visibility",
    "temperature", "dew_point", "pressure"
]

# === Initialize scaler and scale ===
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(df[features])

# === Create scaled DataFrame ===
df_scaled = pd.DataFrame(scaled_values, columns=features)
# === Add cyclical encodings back ===
df_scaled["wind_dir_sin"] = df["wind_dir_sin"]
df_scaled["wind_dir_cos"] = df["wind_dir_cos"]
df_scaled["hour_sin"] = df["hour_sin"]
df_scaled["hour_cos"] = df["hour_cos"]
df_scaled["month_sin"] = df["month_sin"]
df_scaled["month_cos"] = df["month_cos"]
df_scaled["datetime"] = df["datetime"]

# === Scale cyclical features to [0, 1] range ===
df_scaled["wind_dir_sin"] = (df["wind_dir_sin"] + 1) / 2  # Scale to [0, 1] range
df_scaled["wind_dir_cos"] = (df["wind_dir_cos"] + 1) / 2  # Scale to [0, 1] range
df_scaled["hour_sin"] = (df["hour_sin"] + 1) / 2  # Scale to [0, 1] range
df_scaled["hour_cos"] = (df["hour_cos"] + 1) / 2  # Scale to [0, 1] range
df_scaled["month_sin"] = (df["month_sin"] + 1) / 2  # Scale to [0, 1] range
df_scaled["month_cos"] = (df["month_cos"] + 1) / 2  # Scale to [0, 1] range

# === Define output directories ===
scaled_data_dir = processed_metar_folder

# === Save the scaled data ===
df_scaled.to_csv(f"{scaled_data_dir}/scaled_metar_for_lstm.csv", index=False)
joblib.dump(scaler, f"{scaled_data_dir}/scaler.save")

print(f"Scaling complete. Saved scaled data as '{scaled_data_dir}/scaled_metar_for_lstm.csv' and scaler as '{scaled_data_dir}/scaler.save'") # Updated print statement