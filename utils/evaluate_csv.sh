#!/usr/bin/env bash
root=/root/distil-whisper
cd $root
source path.sh
cd $root/utils

csv_fpath=/root/distil-whisper/corpus/output/COOLTEST-longform/baseline_0.1298.tsv

# echo "Evaluating original $csv_fpath"
# python evaluation.py \
#   --csv_fpath $csv_fpath \

# echo "Evaluating $csv_fpath to_simplified"
# python evaluation.py \
#   --csv_fpath $csv_fpath \
#   --to_simplified

# echo "Evaluating $csv_fpath to_traditional"
# python evaluation.py \
#   --csv_fpath $csv_fpath \
#   --to_traditional
for csv_fpath in $(ls /root/distil-whisper/corpus/output/COOLTEST-longform/*.tsv); do
  echo "Evaluating $csv_fpath"
  python evaluation.py \
    --csv_fpath $csv_fpath \
    --to_simplified \
    --calculate_complete_mer
  echo ""
done
# echo "Evaluating separate_language with $csv_fpath"
# python evaluation.py \
#   --csv_fpath $csv_fpath \
#   --to_simplified \
#   --calculate_complete_mer
  # --separate_language \
  # --test_only \

