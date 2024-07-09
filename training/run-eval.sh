#!/usr/bin/env bash
root=~/distil-whisper
cd $root
source path.sh
cd $root/training
export CUDA_VISIBLE_DEVICES=0
model_output_dir=~/distil-whisper/corpus/output

model_save_dir="openai/whisper-base"
# model_save_dir="openai/whisper-small"
# model_save_dir="openai/whisper-medium"
# model_save_dir="openai/whisper-large-v2"
# model_save_dir="openai/whisper-large-v3"

# model_save_dir=$model_output_dir/baseline
# model_save_dir=$model_output_dir/hallucination/6gram_5repeat_detector

# model_save_dir=$model_output_dir/hallucination/whisper-base_threshold0.4
# model_save_dir=$model_output_dir/hallucination/whisper-base_threshold0.6

# model_save_dir=$model_output_dir/hallucination/whisper-base_threshold0.4-phonemized
# model_save_dir=$model_output_dir/hallucination/whisper-base_threshold0.6-phonemized

# model_save_dir=$model_output_dir/hallucination/6gram-5repeat_base-per0.4_allow-empty
# model_save_dir=$model_output_dir/hallucination/6gram-5repeat_base-per0.6_allow-empty

# wandb_name="test_new_machine_NTUML2021-longform"
# wandb_name="test_base_bt64"
# wandb_name="large-small_ASCEND-longform"
# wandb_name="large-v2-mixed-emb_ASCEND-shortform-max64"
# wandb_name="large-v3-mixed-emb-prompt_NTUCOOLTEST-longform"

# wandb_name="baseline_cv-shortform-max64"
# wandb_name="HD-6gram-5repeat-step120k_cv-shortform-max64"

# wandb_name="HD-whisper-base-threshold0.4-steps120k_cv-shortform-max64"
# wandb_name="HD-whisper-base-threshold0.6-steps120k_cv-shortform-max64"

# wandb_name="HD-whisper-base-threshold0.4-phonemized-steps115k_cv-shortform-max64"
# wandb_name="HD-whisper-base-threshold0.6-phonemized-steps115k_cv-shortform-max64"

# wandb_name="HD-6gram-5repeat_whisper-base-per0.4_allow-empty-vbest_cv-shortform-max64"
# wandb_name="HD-6gram-5repeat_whisper-base-per0.6_allow-empty-vbest_cv-shortform-max64"

# dataset_name="CAiRE/ASCEND"
# dataset_name="ky552/ML2021_ASR_ST"
# dataset_name="ky552/cszs_zh_en"
# dataset_name="mozilla-foundation/common_voice_16_1"
# dataset_name="/mnt/home/andybi7676/distil-whisper/data/common_voice_16_1_zh-TW_pseudo_labelled"
# dataset_name="/mnt/home/andybi7676/distil-whisper/data/cool_asr"
dataset_name="/root/distil-whisper/corpus/data/cool_asr_longform"
# dataset_name="/root/distil-whisper/corpus/data/ASCEND_longform"
# dataset_name="/mnt/home/andybi7676/distil-whisper/data/cszs_longform"
# dataset_name="/root/distil-whisper/corpus/data/ntuml2021_longform"
# dataset_config_name="main" # for ASCEND
dataset_config_name="default"
# dataset_config_name="zh-TW"
dataset_split_name="test"
text_column_name="text" # text for ntu cool asr, transcription for ntuml2021, and sentence for common voice
# text_column_name="transcription" # text for ntu cool asr, transcription for ntuml2021 and ASCEND, and sentence for common voice
# text_column_name="sentence" # text for ntu cool asr, transcription for ntuml2021, and sentence for common voice
# text_column_name="correct_transcription" # for cszs only
# audio_column_name="correct_audio" # for cszs only
audio_column_name="audio"

# python run_eval.py \
#   --model_name_or_path $model_save_dir \
#   --wandb_project "speed_test" \
#   --wandb_name ${wandb_name} \
#   --dataset_name $dataset_name \
#   --dataset_config_name $dataset_config_name \
#   --dataset_split_name $dataset_split_name \
#   --text_column_name $text_column_name \
#   --audio_column_name $audio_column_name \
#   --batch_size 32 \
#   --dtype "float16" \
#   --generation_max_length 256 \
#   --language "zh" \
#   --attn_implementation "sdpa" \
#   --mix_lang_emb True 

for i in 128 64 32 16 8 4; do
  for model_name in base small medium large-v2; do
    # model_save_dir="openai/whisper-small"
    # model_save_dir="openai/whisper-medium"
    # model_save_dir="openai/whisper-large-v2"
    model_save_dir="openai/whisper-$model_name"
    wandb_name="test_${model_name}_bs$i"
    python run_eval.py \
      --model_name_or_path $model_save_dir \
      --wandb_project "inference_speed_test" \
      --wandb_name ${wandb_name}_bs${i} \
      --dataset_name $dataset_name \
      --dataset_config_name $dataset_config_name \
      --dataset_split_name $dataset_split_name \
      --text_column_name $text_column_name \
      --audio_column_name $audio_column_name \
      --batch_size $i \
      --dtype "float16" \
      --generation_max_length 256 \
      --language "zh" \
      --attn_implementation "sdpa" \
      --mix_lang_emb True 
      # --dataset_cache_dir $dataset_name 
      # --use_pipeline True 
      # --prompt_text "This is a code-switching sentence. Transcribe it." 
      # --streaming 
  done
done