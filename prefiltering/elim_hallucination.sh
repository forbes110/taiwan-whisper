python elim_hallucination.py \
    --original_tsv /mnt/audio_paths.tsv \
    --type whisper \
    --hyps_txt /mnt/validator_inference.txt \
    --output_dir /mnt \
    --threshold 0.4 \
    --num_workers 16 