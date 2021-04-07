#!/usr/bin/env bash

PWD=$(pwd)
RAW='data/raw'
PREPROC='data/preprocessed'
FILES=($(ls ${RAW}))

for FILE in ${FILES[@]}; do
    echo "Processing $FILE"
    awk '{print $1, $NF}' $PWD/$RAW/$FILE > $PWD/$PREPROC/$FILE
done

echo "Done."
