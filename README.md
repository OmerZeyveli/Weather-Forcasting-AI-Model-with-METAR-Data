# 🌤️ Weather Prediction with METAR + LSTM
This project focuses on enhancing **short-term weather forecasting** at airfields using **METAR (Meteorological Aerodrome Reports)** and **Long Short-Term Memory (LSTM)** neural networks.

Accurate forecasting is crucial for flight safety, especially for student pilots, whose training heavily depends on **calm and clear weather conditions.** Using structured hourly METAR reports, this project builds a data-driven machine learning pipeline to predict:

- Surface wind direction and speed
- Visibility
- Temperature and dew point
- Atmospheric pressure

&nbsp;

## 📂 Project Folder Structure


```
demo/
├── metar_raw/
│   └── raw_metar.csv            # Your input METAR data
├── metar_processed/             # Stores parsed & scaled data
├── input_splits/                # Train/val/test data splits
└── model_weights/               # Saved model and scaler
```

&nbsp;

## 🚀 Quick Start

**1. Set up Environment**
```bash 
pip install -r requirements.txt
```

&nbsp;

**2. Prepare Input Data**

Place your METAR data as "raw_metar.csv" inside:
```
demo/metar_raw/raw_metar.csv
```

The file should contain at least these columns:
- valid: Timestamp of report
- metar: Raw METAR string

&nbsp;

**3. Run the Full Pipeline**
```bash 
bash Train.sh
```

This script will:

1. Parse raw METAR reports -> structured CSV
2. Scale (normalize) and encode data
3. Create sequences for LSTM input/output
4. Train the LSTM model
5. Evaluate the model on the test set

&nbsp;

## 📊 Model Details
- Architecture: 2-layer LSTM (128 + 64 units) + Dense output

- Input: Past 168 hours of METAR-derived weather data

- Output: 12-hour prediction for:
    - Wind direction (as sin/cos)
    - Wind speed
    - Visibility
    - Temperature
    - Dew point
    - Pressure
    - Wind direction variability

## 📬 Contact
Questions or suggestions? Feel free to open an issue or contribute via pull request.