# parser = argparse.ArgumentParser(description="Transcribe audio files using WhisperModel.")
# parser.add_argument("--dataset_path", type=str, default="/mnt/dataset_1T/tmp_dir/sample.tsv", help="Path to dataset.csv file.")
# parser.add_argument("--output_dir", type=str, default="/mnt/pseudo_label", help="Directory to save output CSV files.")
# parser.add_argument("--language", type=str, default="zh", help="Language code for transcription.")
# parser.add_argument("--log_progress", default=True, help="Display progress bars during transcription.")
# parser.add_argument("--model_size_or_path", type=str, default="tiny", help="Size or path of the Whisper model.")
# parser.add_argument("--compute_type", type=str, default="default", help="Compute type for CTranslate2 model.")
# parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for parallel processing.")

python3 inference.py \
    --dataset_path /mnt/dataset_1T/tmp_dir/sample.tsv \
    --output_dir /mnt/pseudo_label \
    --language zh \
    --log_progress True \
    --model_size_or_path tiny \
    --compute_type default \
    --num_workers 8