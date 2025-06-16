#!/bin/bash
set -e

# put your raw_metar.csv data in the demo/metar_raw/ folder
prefix="demo/"

# metar_raw should has null values for missing data
python Parse.py ${prefix}
python Scale.py ${prefix}
python Split.py ${prefix}

# If you want to train the model again, you can comment out the Parse, Scale, and Split steps
python Train.py ${prefix}

# If you only want to test the model, you can comment out the Train and other steps
python Test.py ${prefix}
python Test_Scaled.py ${prefix}