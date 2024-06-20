root=~/distil-whisper
data_root=~/distil-whisper/corpus/data

cd $root
source path.sh
cd $root/training

accelerate launch --main_process_port 29500 run_distillation.py \
  --model_name_or_path "$root/student_models/basic" \
  --teacher_model_name_or_path "openai/whisper-large-v2" \
  --train_dataset_manifest "$data_root/manifest/train-all.tsv" \
  --train_dataset_root $data_root/train/categorized \
  --train_dataset_name "" \
  --train_split_name "" \
  --text_column_name "" \
  --train_dataset_samples "" \
  --eval_dataset_name "$root/data/common_voice_16_1_zh-TW_pseudo_labelled" \
  --eval_split_name "test" \
  --eval_text_column_name "sentence" \
  --eval_steps 1000 \
  --save_steps 5000 \
  --warmup_steps 50 \
  --learning_rate 0.0001 \
  --lr_scheduler_type "constant_with_warmup" \
  --timestamp_probability 0.5 \
  --condition_on_prev_probability 0.2 \
  --language "zh" \
  --task "transcribe" \
  --logging_steps 100 \
  --save_total_limit 20 \
  --max_steps 120000 \
  --wer_threshold 20 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 2 \
  --dataloader_num_workers 32 \
  --preprocessing_num_workers 16 \
  --ddp_timeout 7200 \
  --dtype "bfloat16" \
  --attn_implementation "sdpa" \
  --output_dir "./output/hallucination/6gram_5repeat_detector" \
  --do_train \
  --do_eval \
  --gradient_checkpointing \
  --overwrite_output_dir \
  --predict_with_generate \
  --freeze_encoder \
  --freeze_embed_positions \
  --streaming True \
  --mix_lang_emb True