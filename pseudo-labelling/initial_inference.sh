
# TODO: here need to be large-v2
python3 initial_inference.py \
    --dataset_path /mnt/dataset_1T/tmp_dir/sample.tsv \
    --output_dir /mnt/pseudo_label \
    --language zh \
    --log_progress True \
    --model_size_or_path tiny \
    --compute_type default \
    --chunk_length 5 \
    --num_workers 8 \
    --batch_size 8