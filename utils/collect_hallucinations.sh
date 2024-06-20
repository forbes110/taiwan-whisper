#!/usr/bin/env bash
root=/mnt/home/andybi7676/distil-whisper
cd $root
source path.sh
cd $root/utils

python collect_hallucinations.py \
    --non_hallucinated_tsv /mnt/home/andybi7676/distil-whisper/corpus/data/manifest/mix_detection/allow_empty/train_non-hallucinated-whisper-base-threshold0.4-phonemized-mix_detection-allow_empty.tsv \
    --original_tsv /mnt/home/andybi7676/distil-whisper/corpus/data/manifest/train-all_new.tsv \
    --small_model_trans_tsv /mnt/home/andybi7676/distil-whisper/dataset/inference_results/whisper_base/idx_hyp.txt \
    --output_dir /mnt/home/andybi7676/distil-whisper/corpus/data/analysis/hallucinations/6gram-5repeat_base-per0.4_allow-empty