#!/usr/bin/env bash
root=~/distil-whisper
cd $root
source path.sh
cd $root/utils
CUDA_VISIBLE_DEVICES=1

# model_id=/root/distil-whisper/corpus/output/baseline
# model_id=/root/distil-whisper/corpus/output/hallucination/6gram_5repeat_detector

# model_id=/root/distil-whisper/corpus/output/hallucination/whisper-base_threshold0.4
# model_id=/root/distil-whisper/corpus/output/hallucination/whisper-base_threshold0.6

# model_id=/root/distil-whisper/corpus/output/hallucination/whisper-base_threshold0.4-phonemized
# model_id=/root/distil-whisper/corpus/output/hallucination/whisper-base_threshold0.6-phonemized

# model_id=/root/distil-whisper/corpus/output/hallucination/6gram-5repeat_base-per0.4_allow-empty
# model_id=/root/distil-whisper/corpus/output/hallucination/6gram-5repeat_base-per0.6_allow-empty

# model_id=openai/whisper-large-v2
# model_id=openai/whisper-medium
model_id=openai/whisper-small
testing_data_root=/root/distil-whisper/corpus/ntu_cool/data/test/for_longform

for lang in zh; do
    python longform_eval.py \
        --normalize \
        --model_id $model_id \
        --testing_data_root $testing_data_root \
        --device_id $CUDA_VISIBLE_DEVICES \
        --language $lang 
        # --mix_lang_emb
done
