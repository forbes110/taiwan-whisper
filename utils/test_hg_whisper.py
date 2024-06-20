from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizerFast
from datasets import load_dataset, Audio
from utils.model_utils import mix_language_embeddings
import soundfile as sf
import torch
import numpy as np

whisper_model_card = "openai/whisper-large-v2"
model = WhisperForConditionalGeneration.from_pretrained(whisper_model_card, low_cpu_mem_usage=True)
fast_tokenizer = WhisperTokenizerFast.from_pretrained(
        whisper_model_card,
        use_fast=True,
    )
# processor = WhisperProcessor.from_pretrained(whisper_model_card)
# forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe", no_timestamps=False)
# print(forced_decoder_ids)
# forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="transcribe", no_timestamps=False)
# print(forced_decoder_ids)
fast_tokenizer.set_prefix_tokens(language="chinese", task="transcribe", predict_timestamps=True)
encode_with_specials = fast_tokenizer(" ", add_special_tokens=True)
encode_wo_specials = fast_tokenizer(" ", add_special_tokens=False)
print(encode_with_specials)
print(encode_wo_specials)
# print(fast_tokenizer.decode([50361, 50364, 3581, 5155, 47935, 2437, 115, 32681, 50464, 50514, 12022, 28455, 34127, 11319, 28533, 21686, 24686, 14475, 50714, 50764, 5884, 29979, 18144, 17144, 43362, 36379, 31217, 10439, 23756, 50914, 50964, 12560, 20545, 11957, 107, 51064, 51114, 46225, 15789, 24293, 10928, 4622, 11100, 2289, 20682, 28533, 1546, 3582, 7322, 51314, 51364, 8949, 47402, 27408, 39517, 51464, 51514, 36897, 19517, 238, 36473, 51614], decode_with_timestamps=True, skip_special_tokens=False))
print(fast_tokenizer.decode(encode_with_specials["input_ids"], skip_special_tokens=False))
print(fast_tokenizer.decode(encode_wo_specials["input_ids"], skip_special_tokens=False))
# print(processor.tokenizer.get_vocab())
# print(processor.tokenizer.encoder)
# print(processor.tokenizer.__class__)
# value = processor.tokenizer.encoder['']
# del processor.tokenizer.encoder['']
# processor.tokenizer.encoder['<|continue|>'] = value
# processor.tokenizer.decoder[value] = "<|continue|>"
# test_text = "<|continue|> 你好"
# ids = processor.tokenizer.encode(test_text)
# print(ids)
# print(processor.tokenizer.decode(ids))
# print(fast_tokenizer.__class__)
# print(fast_tokenizer.get_vocab())
# fast_tokenizer.add_special_tokens({"additional_special_tokens": ["<|continue|>"]})
# value = fast_tokenizer.encoder['']
# del fast_tokenizer.encoder['']
# fast_tokenizer.encoder['<|continue|>'] = value
# fast_tokenizer.decoder[value] = "<|continue|>"
# test_text = "<|continue|> 你好"
# ids = fast_tokenizer.encode(test_text)
# print(ids)
# print(fast_tokenizer.decode(ids))
# print(forced_decoder_ids)
# ------------test mix language embeddings----------------
# print(model)
# print(forced_decoder_ids)
# zh_id = processor.tokenizer.convert_tokens_to_ids("<|zh|>")
# en_id = processor.tokenizer.convert_tokens_to_ids("<|en|>")
# print("<|en|>: ", en_id) # 50259
# print("<|zh|>: ", zh_id) # 50260
# # mix weight to <|zh|>
# print(model.model.decoder.embed_tokens(torch.tensor([zh_id])))
# print(model.model.decoder.embed_tokens(torch.tensor([en_id])))
# model = mix_language_embeddings(model, processor.tokenizer, languages=['en', 'zh'], target_language='zh', weights=[0.5, 0.5])
# print(model.model.decoder.embed_tokens(torch.tensor([zh_id])))

# print(model.model.decoder.embed_tokens(torch.tensor([zh_id])))
# model.to("cuda")
# ------------test mix language embeddings----------------

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