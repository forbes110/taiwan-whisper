from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
import soundfile as sf
import numpy as np

whisper_model_card = "openai/whisper-large-v2"
model = WhisperForConditionalGeneration.from_pretrained(whisper_model_card, low_cpu_mem_usage=True)
processor = WhisperProcessor.from_pretrained(whisper_model_card)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="transcribe", no_timestamps=False)
# print(processor.tokenizer.get_vocab())
print(forced_decoder_ids)

print(model)

# model.to("cuda")

# input_data_fpath = "/home/andybi7676/distil-whisper/corpus/ntu_cool/data/train/hallucinated/42964/42964_0-480000.flac"
# # input_data_fpath = "/home/andybi7676/distil-whisper/tmp/186998.flac"
# input_data, sr = sf.read(input_data_fpath)
# assert sr == 16000, f"Sampling rate {sr} is not 16kHz"

# inputs = processor(input_data, sampling_rate=16000, return_tensors="pt")
# input_features = inputs.input_features

# generated_ids = model.generate(input_features.to("cuda"), max_new_tokens=400, forced_decoder_ids=forced_decoder_ids, return_timestamps=True)
# pred_text = processor.decode(generated_ids[0], skip_special_tokens=False, decode_with_timestamps=True)

# print("Pred text:", pred_text)


# common_voice = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="validation", streaming=True)
# common_voice = common_voice.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
# print("Environment set up successful?", generated_ids.shape[-1] == 19)