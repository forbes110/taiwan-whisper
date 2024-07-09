#!/usr/bin/env bash
root=/root/distil-whisper
cd $root
source path.sh
cd $root/utils

python hallucination_detector.py \
    --overwrite_root /root/distil-whisper/corpus/ntu_cool/data/train/categorized \
    --original_tsv /root/distil-whisper/corpus/ntu_cool/manifest/train-all_new.tsv \
    --type whisper \
    --hyps_txt /root/distil-whisper/corpus/output/inference_results/whisper_base/idx_hyp.txt \
    --output_dir /root/distil-whisper/corpus/ntu_cool/manifest/mer \
    --threshold 0.8 \
    --num_workers 16 
    # --phonemize \
    # --mix_detection \
    # --empty_error_rate 0.0 \
    # --additional_fname "allow_empty" \