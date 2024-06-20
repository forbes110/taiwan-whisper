#!/usr/bin/env bash
root=/mnt/home/andybi7676/distil-whisper
cd $root
source path.sh
cd $root/utils

python hallucination_detector.py \
    --original_tsv /mnt/home/andybi7676/distil-whisper/corpus/data/manifest/train-all_new.tsv \
    --type whisper \
    --hyps_txt /mnt/home/andybi7676/distil-whisper/dataset/inference_results/whisper_base/idx_hyp.txt \
    --output_dir /mnt/home/andybi7676/distil-whisper/corpus/data/manifest/mix_detection/allow_empty \
    --phonemize \
    --mix_detection \
    --empty_error_rate 0.0 \
    --additional_fname "allow_empty" \
    --threshold 0.6 \
    --num_workers 32 