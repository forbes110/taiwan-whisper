#!/usr/bin/env bash
root=/mnt/home/andybi7676/distil-whisper
cd $root
source path.sh
cd $root/training
CUDA_VISIBLE_DEVICES=0

# model_save_dir="/mnt/home/andybi7676/distil-whisper/training/output/baseline"
# model_save_dir="/mnt/home/andybi7676/distil-whisper/training/output/hallucination/6gram_5repeat_detector"
# model_save_dir="/mnt/home/andybi7676/distil-whisper/training/output/hallucination/whisper-base_threshold0.4"
# model_save_dir="/mnt/home/andybi7676/distil-whisper/training/output/hallucination/whisper-base_threshold0.6"
# model_save_dir="/mnt/home/andybi7676/distil-whisper/training/output/hallucination/6gram-5repeat_base-per0.4_allow-empty"
# model_save_dir="/mnt/home/andybi7676/distil-whisper/training/output/hallucination/6gram-5repeat_base-per0.6_allow-empty"
# model_save_dir="/mnt/home/andybi7676/distil-whisper/training/output/hallucination/whisper-base_threshold0.4-phonemized"
# model_save_dir="/mnt/home/andybi7676/distil-whisper/training/output/hallucination/whisper-base_threshold0.6-phonemized"
model_save_dir="openai/whisper-large-v2"
# model_save_dir="openai/whisper-large-v3"
# wandb_name="baseline_cszs-longform"
# wandb_name="large-v2-mixed-emb_NTUML2021-longform"
# wandb_name="large-v3-mixed-emb-prompt_NTUCOOLTEST-longform"
# wandb_name="HD-whisper-base-threshold0.4-steps120k_cv-shortform"
wandb_name="test_speed_large_model"
# wandb_name="HD-6gram-5repeat-step120k_cszs-longform"
# wandb_name="HD-6gram-5repeat_whisper-base-per0.4_allow-empty-vbest_NTUCOOLTEST-longform"
# wandb_name="HD-6gram-5repeat_whisper-base-per0.6_allow-empty-vbest_NTUCOOLTEST-longform"
# wandb_name="HD-whisper-base-threshold0.6-steps120k_cszs-longform"
# wandb_name="HD-whisper-base-threshold0.4-steps120k_NTUML2021-longform"
# wandb_name="HD-whisper-base-threshold0.6-phonemized-steps115k_NTUML2021-shortform"
# wandb_name="HD-whisper-base-threshold0.4-phonemized-steps115k_cszs-longform"
# wandb_name="HD-whisper-base-threshold0.6-phonemized-steps115k_NTUML2021-longform"

# dataset_name="CAiRE/ASCEND"
# dataset_name="ky552/ML2021_ASR_ST"
# dataset_name="ky552/cszs_zh_en"
# dataset_name="mozilla-foundation/common_voice_16_1"
# dataset_name="/mnt/home/andybi7676/distil-whisper/data/common_voice_16_1_zh-TW_pseudo_labelled"
# dataset_name="/mnt/home/andybi7676/distil-whisper/data/cool_asr"
# dataset_name="/mnt/home/andybi7676/distil-whisper/data/cool_asr_longform"
# dataset_name="/mnt/home/andybi7676/distil-whisper/data/ASCEND_longform"
# dataset_name="/mnt/home/andybi7676/distil-whisper/data/cszs_longform"
dataset_name="/mnt/home/andybi7676/distil-whisper/data/ntuml2021_longform"
# dataset_config_name="main" # for ASCEND
dataset_config_name="default"
# dataset_config_name="zh-TW"
dataset_split_name="test"
# text_column_name="text" # text for ntu cool asr, transcription for ntuml2021, and sentence for common voice
text_column_name="transcription" # text for ntu cool asr, transcription for ntuml2021 and ASCEND, and sentence for common voice
# text_column_name="sentence" # text for ntu cool asr, transcription for ntuml2021, and sentence for common voice
# text_column_name="correct_transcription" # for cszs only
# audio_column_name="correct_audio" # for cszs only
audio_column_name="audio"

python run_eval.py \
  --model_name_or_path $model_save_dir \
  --wandb_name $wandb_name \
  --dataset_name $dataset_name \
  --dataset_config_name $dataset_config_name \
  --dataset_split_name $dataset_split_name \
  --text_column_name $text_column_name \
  --audio_column_name $audio_column_name \
  --batch_size 18 \
  --dtype "bfloat16" \
  --generation_max_length 256 \
  --language "zh" \
  --attn_implementation "sdpa" 
  # --mix_lang_emb True 
  # --use_pipeline True 
  # --prompt_text "This is a code-switching sentence. Transcribe it." 
  # --streaming 