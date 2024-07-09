#!/usr/bin/env bash
root=~/distil-whisper
cd $root
source path.sh
cd $root/training

dataset_name="ky552/ML2021_ASR_ST"
dataset_config_name="default" # for ASCEND
dataset_split_name="test"
text_column_name="transcription" # text for ntu cool asr, transcription for ntuml2021, ASCEND, and sentence for common voice
id_column_name="path"
audio_column_name="audio"
# text_column_name="correct_transcription" # text for ntu cool asr, transcription for ntuml2021, ASCEND, and sentence for common voice
# id_column_name="correct_file"
# audio_column_name="correct_audio"
# audio_column_name="audio"
# id_column_name="path"

accelerate launch --num_processes 2 --main_process_port 29500 run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-large-v2" \
  --dataset_name $dataset_name \
  --dataset_config_name $dataset_config_name \
  --dataset_split_name $dataset_split_name \
  --text_column_name $text_column_name \
  --audio_column_name $audio_column_name \
  --id_column_name $id_column_name \
  --output_dir "$root/corpus/data/ntuml2021_longform" \
  --wandb_project "distil-whisper-labelling" \
  --per_device_eval_batch_size 8 \
  --dtype "float16" \
  --attn_implementation "sdpa" \
  --logging_steps 500 \
  --max_label_length 256 \
  --concatenate_audio \
  --preprocessing_batch_size 500 \
  --preprocessing_num_workers 16 \
  --dataloader_num_workers 8 \
  --report_to "wandb" \
  --language "zh" \
  --task "transcribe" \
  --return_timestamps \
  --streaming False \
  --generation_num_beams 1 \
  --mix_lang_emb True