#!/bin/bash
for filename in `ls ../models/best/*.hdf5 | sort -r`; do
    echo "tuning model $filename"
    ./fine_tuning.py $filename
done
