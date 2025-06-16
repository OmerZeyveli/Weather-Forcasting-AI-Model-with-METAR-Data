#!/bin/bash
set -e

prefix="demo/"

python Parse.py ${prefix}
python Scale.py ${prefix}
python Split.py ${prefix}
python Train.py ${prefix}
python Test.py ${prefix}