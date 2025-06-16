#!/bin/bash
set -e

prefix="demo/"

# Path to your "raw_metar.csv" file
metar_raw_folder="${prefix}metar_raw/"

metar_processed_folder="${prefix}metar_processed/"
input_splits_folder="${prefix}input_splits/"
model_weights_folder="${prefix}model_weights/"

# Create directories if they don't exist
mkdir -p ${metar_raw_folder}
mkdir -p ${metar_processed_folder}
mkdir -p ${input_splits_folder}
mkdir -p ${model_weights_folder}

# metar_raw should has null values for missing data
python Parse.py ${metar_raw_folder} ${metar_processed_folder}
python Scale.py ${metar_processed_folder}
python Split.py ${metar_processed_folder} ${input_splits_folder} 0.2 0.1 # Adjust the split ratios as needed (e.g., 0.2 for test, 0.1 for validation)

# If you want to train the model again, you can comment out the Parse, Scale, and Split steps
python Train.py ${input_splits_folder} ${model_weights_folder}

# If you only want to test the model, you can comment out the Train and other steps
python Test.py ${model_weights_folder} ${input_splits_folder} ${metar_processed_folder}
python Test_Scaled.py ${model_weights_folder} ${input_splits_folder}