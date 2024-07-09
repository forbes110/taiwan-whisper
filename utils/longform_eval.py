import torch
import glob
import os
import os.path as osp
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm
from utils.evaluation import MixErrorRate
from utils.transcript_readers import read_vtt
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from utils.model_utils import mix_language_embeddings

def eval_longform_single(audio_fpath, trans_fpath, pipe, normalizer=None):
    segments = read_vtt(trans_fpath)
    ref = " ".join([seg[-1] for seg in segments])
    output = pipe(audio_fpath)
    # print(output)
    hyp = " ".join(chunk["text"] for chunk in output["chunks"])
    # print(hyp)
    if normalizer:
        hyp = normalizer(hyp)
        ref = normalizer(ref)
    return hyp, ref

def main(args):
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = args.model_id

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    processor = AutoProcessor.from_pretrained(model_id)
    if args.mix_lang_emb:
        print("Mixing language embeddings...")
        if not "large" in model_id:
            raise ValueError("Language embedding mixing is only supported for large models.")
        zh_emb = model.model.decoder.embed_tokens.weight[50260]
        en_emb = model.model.decoder.embed_tokens.weight[50259]
        print(f"zh: {zh_emb}, {zh_emb.shape}")
        print(f"en: {en_emb}, {en_emb.shape}")
        model = mix_language_embeddings(model, processor.tokenizer, languages=['zh', 'en'])
    model.to(device)
    model.eval()
    # print(model)
    zh_emb = model.model.decoder.embed_tokens.weight[50260]
    en_emb = model.model.decoder.embed_tokens.weight[50259]
    print(f"zh: {zh_emb}, {zh_emb.shape}")
    print(f"en: {en_emb}, {en_emb.shape}")

    generate_kwargs = {
        'language': args.language,
        'task': 'transcribe',
        'return_timestamps': True,
    }
    model.generation_config.update(**generate_kwargs)
    processor.tokenizer.set_prefix_tokens(language=args.language, task="transcribe", predict_timestamps=True)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps=True,
        generate_kwargs=generate_kwargs,
        max_new_tokens=256,
        torch_dtype=torch_dtype,
        device=device,
    )
    testing_data_root = args.testing_data_root
    audio_fpaths = glob.glob(f"{testing_data_root}/**/*.wav", recursive=True)
    trans_fpaths = list(map(lambda x: x.replace(".wav", ".srt"), audio_fpaths))
    if args.test:
        audio_fpaths = audio_fpaths[:2]
        trans_fpaths = trans_fpaths[:2]
    normalizer = None
    if args.normalize:
        normalizer = BasicTextNormalizer()
    hyps, refs = [], []
    for audio_fpath, trans_fpath in tqdm(zip(audio_fpaths, trans_fpaths), total=len(audio_fpaths), desc="Evaluating..."):
        hyp, ref = eval_longform_single(audio_fpath, trans_fpath, pipe, normalizer=normalizer)
        hyps.append(hyp)
        refs.append(ref)
    metric = MixErrorRate()
    mer = metric.compute(hyps, refs)
    print(f"{model_id}: {mer}")
    if "large" in model_id:
        model_id = f"/root/distil-whisper/corpus/output/{model_id}"
        os.makedirs(model_id, exist_ok=True)
    output_dir = osp.join(model_id, "cool_test_real_longform")
    os.makedirs(output_dir, exist_ok=True)
    result_fpath = osp.join(output_dir, f"result_{args.language}_{mer:.4f}.tsv")
    config_output_fpath = osp.join(output_dir, "cool_test_config.txt")
    with open(result_fpath, 'w') as fw:
        print("audio_fpath\thyp\tref", file=fw)
        for hyp, ref, audio_f, trans_f in zip(hyps, refs, audio_fpaths, trans_fpaths):
            print(f"{audio_f}\t{hyp}\t{ref}", file=fw)
    with open(config_output_fpath, 'a') as fw:
        print(f"MER: {mer}", file=fw)
        print(f"Model ID: {model_id}", file=fw)
        print(f"Testing data root: {testing_data_root}", file=fw)
        print(f"Device ID: {args.device_id}", file=fw)
        print(f"Test: {args.test}", file=fw)
        print(f"Normalize: {args.normalize}", file=fw)
        print(f"Mix lang emb: {args.mix_lang_emb}", file=fw)
        print(f"Language: {args.language}", file=fw)
        print(f"Generated kwargs: {generate_kwargs}", file=fw)
    # log zh, en embeddings in np format
    np.save(osp.join(model_id, "zh_emb.npy"), zh_emb.cpu().detach().numpy())
    np.save(osp.join(model_id, "en_emb.npy"), en_emb.cpu().detach().numpy())
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--testing_data_root", type=str, required=True)
    parser.add_argument("--language", type=str, default="zh")
    parser.add_argument("--device_id", default=0, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--mix_lang_emb", action="store_true")
    args = parser.parse_args()

    main(args)
