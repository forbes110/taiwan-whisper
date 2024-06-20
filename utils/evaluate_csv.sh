#!/usr/bin/env bash
root=/mnt/home/andybi7676/distil-whisper
cd $root
source path.sh
cd $root/utils

csv_fpath=/mnt/home/andybi7676/distil-whisper/training/output/baseline/baseline_cv-shortform.csv

echo "Evaluating original $csv_fpath"
python evaluation.py \
  --csv_fpath $csv_fpath \

echo "Evaluating $csv_fpath to_simplified"
python evaluation.py \
  --csv_fpath $csv_fpath \
  --to_simplified

echo "Evaluating $csv_fpath to_traditional"
python evaluation.py \
  --csv_fpath $csv_fpath \
  --to_traditional

