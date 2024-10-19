import os
import os.path as osp
import argparse
import glob
import torch
import re
import numpy as np
import soundfile as sf
import multiprocessing as mp
from tqdm import tqdm
from transformers import (
    AddedToken,
    WhisperForConditionalGeneration, 
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from collections import defaultdict
from utils.transcript_readers import read_vtt, timecode_to_seconds
from utils.evaluation import MixErrorRate
from functools import partial

class WhisperSmallModelDetector(object):
    def __init__(self, small_model_card='openai/whisper-base', accelerator=None):
        self.model = WhisperForConditionalGeneration.from_pretrained(small_model_card, low_cpu_mem_usage=True)
        self.processor = WhisperProcessor.from_pretrained(small_model_card)
        self.tokenizer = WhisperTokenizerFast.from_pretrained(small_model_card, use_fast=True)
        self.accelerator = accelerator
        # override timestamp tokens until tokenizer issues are fixed in transformers --> what issue? 
        timestamps = [AddedToken("<|%.2f|>" % (i * 0.02), lstrip=False, rstrip=False) for i in range(1500 + 1)]
        # with logging.disable(logging.WARNING):
        self.tokenizer.add_tokens(timestamps)
        self.tokenizer.set_prefix_tokens(language='zh', task='transcribe', predict_timestamps=True)
        self.gen_kwargs = {
            "max_length": 448,
            "num_beams": 1,
            "return_timestamps": True,
            "language": 'zh',
            "task": 'transcribe',
        }
        # other kwargs
        self.input_padding = "longest"
        # prepare model
        self.model.eval()
        if self.accelerator is not None:
            self.model = self.model.to(self.accelerator.device)
    
    def collate_fn(self, features):
        # batch keys: 'idx', 'path', 'array'
        inputs = self.processor.feature_extractor(
            [feature['array'] for feature in features],
            sampling_rate=16000,
        )
        # print(inputs.input_features.shape)
        input_features = {
            'input_features': inputs.input_features
        }
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.input_padding,
            return_tensors="pt",
        )
        batch['idx'] = torch.from_numpy(np.array([feature['idx'] for feature in features])).long().to(batch['input_features'].device)
        # batch['additional'] = "hello"
        # print(batch)
        # assert False
        return batch # dict of tensors

    def generate(self, batch):
        with torch.no_grad():
            output_ids = self.model.generate(batch["input_features"], **self.gen_kwargs)
            # output_ids = self.accelerator.pad_across_processes(output_ids, dim=1, pad_index=self.tokenizer.pad_token_id)
            # pad to longest generated sequence
            # output_ids = torch.nn.functional.pad(output_ids, (0, 448 - output_ids.shape[1]), value=self.tokenizer.pad_token_id)
        # print(output_ids.shape)
        preds = output_ids
        preds_str = self.tokenizer.batch_decode(preds, skip_special_tokens=True, decode_with_timestamps=False)
        
        # return [(idx.item(), pred_s) for idx, pred_s in zip(batch['idx'], pred_str)]
        return batch['idx'].cpu().numpy(), preds_str
            

def check_hallucination(segment, **kwargs):
    if type(segment) == str:
        text = segment
    elif type(segment) == tuple:
        s, e, text = segment
    def _length_checker(text, threshold=150):
        if len(text) < threshold:
            return False
        return True

    def _char_ngram_checker(text, n=6, threshold=5):
        ngram_counts = defaultdict(lambda: 0)
        if len(text) < n:
            return False
        for i in range(len(text) - n + 1):
            ngram = text[i:i+n]
            if '|>' in ngram or '<|' in ngram: continue
            ngram_counts[ngram] += 1
        counts = ngram_counts.values()
        counts = np.array(list(counts), dtype=np.float16)
        max_count = counts.max()
        if max_count > threshold:
            return True
        return False
        # prob = counts / counts.sum()
        # if prob.max() > threshold:
        #     return True
    
    # return _length_checker(text, **kwargs)
    return _char_ngram_checker(text, **kwargs)

def check_single_trans(input, skip_special_tokens=True, metric=None, threshold=0.5, normalizer=None, mix_detection=False, empty_error_rate=1.0):
    idx, trans_fpath, hyp = input
    with open(trans_fpath, "r") as f:
        lines = f.readlines()
        whisper_transcript = lines[0].strip().split("<|endoftext|>")[0] # remove <|endoftext|>
        whisper_transcript = whisper_transcript.split("<|continued|>")[0] # remove <|continued|>
        end_transcript = lines[2].strip()
        # find all timestamp tokens ('<|*|>') and remove them in the transcript
        timestamp_tokens = re.findall(r"<\|\d{1,2}\.\d{2}\|>", whisper_transcript + end_transcript)
        for st in timestamp_tokens:
            whisper_transcript = whisper_transcript.replace(st, ' ')
        whisper_transcript = whisper_transcript.strip().replace('  ', ' ')
    if normalizer is not None:
        hyp = normalizer(hyp.strip())
        whisper_transcript = normalizer(whisper_transcript)
    if mix_detection:
        # check by ngram hallucination
        large_is_hallucinated = check_hallucination(whisper_transcript, n=6, threshold=5)
        if large_is_hallucinated:
            return idx, True
        small_is_hallucinated = check_hallucination(hyp, n=6, threshold=5)
        if small_is_hallucinated: # if large is not trivially hallucinated but the small one is, then the small one should not be used to perform examination
            return idx, False
    # check hallucination by comparing the original transcript and the hyp transcript with MER
    mer = metric.compute([whisper_transcript], [hyp], show_progress=False, empty_error_rate=empty_error_rate)
    if mer > threshold:
        return idx, True
    return idx, False

def whisper_checker(
    original_tsv, 
    hyps_tsv, 
    output_dir, 
    threshold=0.6, 
    num_workers=32, 
    phonemize=False, 
    mix_detection=False, 
    empty_error_rate=1.0, 
    additional_fname="", 
    overwrite_root=""):
    
    # load original tsv
    with open(original_tsv, "r") as f:
        root = f.readline().strip()
        if overwrite_root:
            root = overwrite_root
        audio_subfpaths = [l.strip() for l in f.readlines()]
        audio_fpaths = [osp.join(root, audio_subfpath) for audio_subfpath in audio_subfpaths]
        trans_fpaths = [audio_fpath.replace('flac', 'txt') for audio_fpath in audio_fpaths]
        
    # load preds from validator model
    idx_to_src_and_hyp = defaultdict(lambda: None)
    invalid_line_indices = []
    with open(hyps_tsv, "r") as f:
        for i, l in enumerate(f.readlines()):
            items = l.strip().split('\t')
            if len(items) != 2:
                print(f"Invalid line@{i}: {l}")
                invalid_line_indices.append(i)
                continue
            idx, hyp = items
            idx_to_src_and_hyp[int(idx)] = {
                'hyp': hyp,
            }
    
    print(f"Invalid line counts: {len(invalid_line_indices)}, total lines: {i}, indices: {invalid_line_indices}")
    # check the length alignment between the original trans_fpaths and the hyps
    if not len(trans_fpaths) == len(idx_to_src_and_hyp):
        print(f"Length mismatch, trans_fpaths: {len(trans_fpaths)}, hyps: {len(idx_to_src_and_hyp)}")
    # matching idx to src
    for valid_idx in idx_to_src_and_hyp.keys():
        idx_to_src_and_hyp[valid_idx]['src'] = trans_fpaths[valid_idx]
    idx_and_src_and_hyps = [(k, v['src'], v['hyp']) for k, v in idx_to_src_and_hyp.items()]
    metric = MixErrorRate(phonemize=phonemize)
    normalizer = BasicTextNormalizer()
    # check hallucination
    _check_single = partial(check_single_trans, metric=metric, threshold=threshold, normalizer=normalizer, mix_detection=mix_detection, empty_error_rate=empty_error_rate)
    if mix_detection:
        print("Mixing detection method...")
    def check_chunk(q, chunk, process_idx, progress_bar_mod=1):
        chunk_results = []
        progress_bar = (
            tqdm(total=len(chunk), desc=f"Checking hallucination@Process{process_idx}...", position=process_idx) 
            if process_idx % progress_bar_mod == 0 
            else None
        )
        for x in chunk:
            res = _check_single(x)
            chunk_results.append(res)
            if progress_bar is not None:
                progress_bar.update(1)
        q.put(chunk_results)
    # # check hallucination using multiprocessing
    # with mp.Pool() as pool:
    #     # show progress bar and check hallucination
    #     hallucinated_indices = list(tqdm(pool.map(single_checker, idx_and_src_and_hyps), total=len(idx_and_src_and_hyps), desc="Checking hallucination..."))
    hallucinated_indices = []
    def _spread_through_processes(n_processes, idx_and_src_and_hyps):
        n = len(idx_and_src_and_hyps)
        n_per_process = n // n_processes
        for i in range(0, n, n_per_process):
            yield idx_and_src_and_hyps[i:i+n_per_process]
    # overall_progress = tqdm(total=len(idx_and_src_and_hyps), desc="Checking hallucination...", position=0)
    queues, processes = [], []
    for proc_i, idx_and_src_and_hyps_chunk in enumerate(_spread_through_processes(num_workers, idx_and_src_and_hyps)):
        q = mp.Queue()
        queues.append(q)
        p = mp.Process(
            target=check_chunk, 
            args=(q, idx_and_src_and_hyps_chunk, proc_i),
            kwargs={'progress_bar_mod': 8}
        )
        p.start()
        processes.append(p)
    for q in queues:
        hallucinated_indices.extend(q.get())
    for p in processes:
        p.join()


    # for i, (idx, src, hyp) in enumerate(tqdm(idx_and_src_and_hyps, total=len(idx_and_src_and_hyps), desc="Checking hallucination...")):
    #     idx_and_hallucinated = check_single((idx, src, hyp))
    #     hallucinated_indices.append(idx_and_hallucinated)
    # show the statistics
    hallucinated_only = [x[1] for x in hallucinated_indices]
    print(f"Total hallucinated segments: {sum(hallucinated_only)}")
    print(f"Hallucination ratio: {sum(hallucinated_only) / len(hallucinated_indices)}")
    
    # save the new tsv filtered by the hallucinated indices
    
    if output_dir is None:
        # do not save the hallucinated tsv, just return
        print("No output directory specified, not saving the hallucinated tsv...")
        return
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_base_fname = f"train_non-hallucinated-whisper-base-threshold{threshold}"
    if phonemize:
        output_base_fname += "-phonemized"
    if mix_detection:
        output_base_fname += "-mix_detection"
    if additional_fname:
        output_base_fname += f"-{additional_fname}"
        
    output_fname = f"{output_base_fname}.tsv"
    with open(osp.join(output_dir, output_fname), "w") as fw:
        print(root, file=fw)
        for i, (idx, hallucinated) in enumerate(tqdm(hallucinated_indices, total=len(hallucinated_indices), desc="Writing hallucinated tsv...")):
            if not hallucinated:
                print(audio_subfpaths[idx], file=fw)
        


def main(args):
    print(args)
    if args.type == "whisper":
        whisper_checker(args.original_tsv, args.hyps_txt, args.output_dir, 
            threshold=args.threshold, 
            num_workers=args.num_workers, 
            phonemize=args.phonemize,
            mix_detection=args.mix_detection,
            empty_error_rate=args.empty_error_rate,
            additional_fname=args.additional_fname,
            overwrite_root=args.overwrite_root
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_tsv",  # original tsv for filtering
        default="", 
        help="the directory containing the transcripts",
    )
    parser.add_argument(
        "--type",
        default="ngram" # ngram, length, or whisper
    )
    parser.add_argument(
        "--hyps_txt", # required for whisper-smaller-model-based hallucination detection
        default="",
    )
    parser.add_argument(
        "--output_dir",
        default=None
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float
    )
    parser.add_argument(
        "--num_workers",
        default=32,
        type=int
    )
    parser.add_argument(
        "--phonemize",
        action="store_true",
    )
    parser.add_argument(
        "--mix_detection",
        action="store_true",
        help="If true, use the mixing hallucination detection method"
    )
    parser.add_argument(
        "--empty_error_rate",
        default=1.0,
        type=float,
        help="If no reference found, return this error rate instead"
    )
    parser.add_argument(
        "--additional_fname",
        default="",
        help="Additional filename for the output tsv"
    )
    parser.add_argument(
        "--overwrite_root",
        default="",
        help="Overwrite the root directory in the original tsv"
    )
    args = parser.parse_args()

    main(args)