#!/bin/bash
for filename in ../models/best/*.hdf5; do
    echo "tuning model $filename"
    ./fine_tuning.py $filename
done
