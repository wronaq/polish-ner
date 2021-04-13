#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "Expecting path to training config file."
    exit 1
fi


export PYTHONPATH=$(pwd)
python training/run_experiment.py $1


