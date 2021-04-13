#!/usr/bin/env bash

if [ $# -lt 3 ]; then
    echo "Expecting: "
    echo "1) path to training config file,"
    echo "2) path to file with model weights,"
    echo "3) text to predict ner tags (in double quotes),"
    echo "4) flag --gpu (optional)."
    exit 1
fi


export PYTHONPATH=$(pwd)
python polish_ner/predictor.py $1 $2 $3 $4


