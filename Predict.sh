#!/bin/bash
set -e

prefix="demo/"

# Path to your "raw_metar.csv" file
metar_raw_folder="${prefix}metar_raw/"

metar_processed_folder="${prefix}predict/metar_processed/"
input_splits_folder="${prefix}predict/input_splits/"

# This code expects the model weights to be in the demo/model_weights/ folder
model_weights_folder="${prefix}model_weights/"

# Create directories if they don't exist
mkdir -p ${metar_raw_folder}
mkdir -p ${metar_processed_folder}
mkdir -p ${input_splits_folder}

# metar_raw should has null values for missing data
python Parse.py ${metar_raw_folder} ${metar_processed_folder}
python Scale.py ${metar_processed_folder}
python Split.py ${metar_processed_folder} ${input_splits_folder} 1.0 0.0 # Adjusted for testing only (1.0 for test, 0.0 for validation)

python Predict.py ${model_weights_folder} ${input_splits_folder} ${metar_processed_folder}
