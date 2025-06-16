import pandas as pd
from metar import Metar
import numpy as np
import os
import sys

# Load your CSV
prefix = sys.argv[1]  # Get prefix from command line
df = pd.read_csv(f"{prefix}metar_raw/LTBU_withnulls.csv")

# Storage for processed rows
parsed_data = []

def safe_val(val, default=np.nan):
    return val.value() if val is not None else default

# Loop through METARs
for i, row in df.iterrows():
    timestamp = pd.to_datetime(row["valid"])
    metar_str = row["metar"]

    try:
        report = Metar.Metar(metar_str)
        raw_wind_dir = report.wind_dir
        wind_speed = safe_val(report.wind_speed, 0)

        # === New logic for wind direction and variability ===
        if raw_wind_dir is None:
            wind_dir = np.nan
            wind_dir_variable = 1 if "VRB" in metar_str else 0
        elif raw_wind_dir.value() is None:
            wind_dir = np.nan
            wind_dir_variable = 1
        else:
            wind_dir = raw_wind_dir.value()
            wind_dir_variable = 1 if "VRB" in metar_str else 0  # fallback check for VRB

        visibility = safe_val(report.vis, 10000)
        temperature = safe_val(report.temp)
        dew_point = safe_val(report.dewpt)
        pressure = safe_val(report.press)
        # Wind gust is not used because of its sparsity

        parsed_data.append({
            "datetime": timestamp,
            "wind_dir": wind_dir,
            "wind_dir_variable": wind_dir_variable,
            "wind_speed": wind_speed,
            "visibility": visibility,
            "temperature": temperature,
            "dew_point": dew_point,
            "pressure": pressure,
            "hour": timestamp.hour,
            "month": timestamp.month,
        })

    except Exception as e:
        print(f"Skipping row {i}: {e}")
        continue

# Final structured DataFrame
df_parsed = pd.DataFrame(parsed_data)
df_parsed = df_parsed.sort_values("datetime").reset_index(drop=True)

# Save to CSV
output_dir = f"{prefix}metar_processed"
os.makedirs(output_dir, exist_ok=True)
df_parsed.to_csv(f"{output_dir}/processed_metar_for_lstm.csv", index=False)
print(df_parsed.head())
