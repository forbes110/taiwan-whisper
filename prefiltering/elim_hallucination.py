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
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from collections import defaultdict
from transcript_readers import read_vtt, timecode_to_seconds
from evaluation import MixErrorRate
from functools import partial
import csv
from contextlib import contextmanager


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

    return _char_ngram_checker(text, **kwargs)

def check_single_trans(input, skip_special_tokens=True, metric=None, threshold=0.5, normalizer=None, mix_detection=False, empty_error_rate=1.0):
    idx, trans_fpath, hyp = input
    
    with open(trans_fpath, "r") as f:
        lines = f.readlines()
        
        # remove <|endoftext|>
        whisper_transcript = lines[0].strip().split("<|endoftext|>")[0]
        
        # remove <|continued|>
        whisper_transcript = whisper_transcript.split("<|continued|>")[0] 
        
        # prev segment as prompt, but not used here
        end_transcript = lines[1].strip()
        
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
        
        # if large is not trivially hallucinated but the small one is, then the small one should not be used to perform examination
        small_is_hallucinated = check_hallucination(hyp, n=6, threshold=5)
        if small_is_hallucinated: 
            return idx, False
        
    # check hallucination by comparing the original transcript and the valiadtor inferenced hyp transcript with MER
    mer = metric.compute([whisper_transcript], [hyp], show_progress=False, empty_error_rate=empty_error_rate)
    
    if mer > threshold:
        return idx, True, (mer, whisper_transcript, hyp, trans_fpath, idx)
    
    return idx, False, None

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
        
    # load original tsv for audio paths
    with open(original_tsv, "r") as f:
        root = f.readline().strip()
        if overwrite_root:
            root = overwrite_root
        audio_subfpaths = [l.strip() for l in f.readlines()]
        audio_fpaths = [osp.join(root, audio_subfpath) for audio_subfpath in audio_subfpaths]
        trans_fpaths = [audio_fpath.replace('flac', 'txt') for audio_fpath in audio_fpaths]
        
    # load inference from validator model
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
        
    hallucinated_indices = []
    hallucination_info = []
    
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
        
    # Collect results from all processes
    for q in queues:
        results = q.get()
        for idx, is_hallucinated, info in results:
            hallucinated_indices.append((idx, is_hallucinated))
            if info is not None:
                hallucination_info.append(info)   
        
    for p in processes:
        p.join()

    # show the statistics
    hallucinated_only = [x[1] for x in hallucinated_indices]
    
    # Write results to CSV after all processes complete
    with open(f'{output_dir}/hallucination_result.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['MER', 'Initial_Inference', 'Validator_Inference', 'Initial_Inference_Path', 'Index'])
        for info in hallucination_info:
            writer.writerow(info)
    
    print(f"Total hallucinated segments: {sum(hallucinated_only)}")
    print(f"Hallucination ratio: {sum(hallucinated_only) / len(hallucinated_indices)}")
    
    # save the new tsv filtered by the hallucinated indices
    if output_dir is None:
        # do not save the hallucinated tsv, just return
        print("No output directory specified, not saving the hallucinated tsv...")
        return
    
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    output_base_fname = f"cleaned-threshold-{threshold}"
    
    if phonemize:
        output_base_fname += "-phonemized"
    if mix_detection:
        output_base_fname += "-mix_detection"
    if additional_fname:
        output_base_fname += f"-{additional_fname}"
    
    # output cleaned dataset    
    output_fname = f"{output_base_fname}.tsv"
    
    # return cleaned dataset
    with open(osp.join(output_dir, output_fname), "w") as fw:
        print(root, file=fw)
        for i, (idx, hallucinated) in enumerate(tqdm(hallucinated_indices, total=len(hallucinated_indices), desc="Writing cleaned tsv...")):
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
            overwrite_root=args.overwrite_root,
        )
    print("Everthing is done!")


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

