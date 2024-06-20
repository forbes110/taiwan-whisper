# collect hallucinations based on the hallucination detection results csv and the original csv
import re
import os
import os.path as osp
import csv 
import random
import argparse
import shutil
from collections import defaultdict

def collect_hallucinations(non_hallucinated_tsv, original_tsv, small_model_trans_tsv, output_dir, num_samples=1000, seed=0):
    original_audio_fpaths = []
    hallucinations_audio_fpaths = []
    non_hallucinations_markers_dict = defaultdict(lambda: False)
    root = None
    # read the original csv
    with open(original_tsv, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        root = next(reader)[0].strip()
        for row in reader:
            original_audio_fpaths.append(row[0])
    # read the hallucination_detection_csv
    with open(non_hallucinated_tsv, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        _root = next(reader)[0]
        for row in reader:
            non_hallucinations_markers_dict[row[0]] = True
    # collect hallucinations
    for i, audio_fpath in enumerate(original_audio_fpaths):
        if not non_hallucinations_markers_dict[audio_fpath]:
            hallucinations_audio_fpaths.append((i, audio_fpath))
    # get the small model text
    small_model_trans_dict = defaultdict(lambda: "")
    with open(small_model_trans_tsv, 'r') as f:
        for i, l in enumerate(f.readlines()):
            items = l.strip().split('\t')
            if len(items) != 2:
                print(f"Invalid line@{i}: {l}")
                continue
            idx, hyp = items
            small_model_trans_dict[int(idx)] = hyp
    # write to output_dir
    random.seed(seed)
    random.shuffle(hallucinations_audio_fpaths)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(osp.join(output_dir, "audio_samples"), exist_ok=True)
    with open(osp.join(output_dir, f"hallucinations_ex{num_samples}_seed{seed}.csv"), 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["index_in_origin", "audio_fpath", "trans_text", "small_model_trans_text"])
        rows = []
        for idx, audio_fpath in hallucinations_audio_fpaths[:num_samples]:
            trans_fpath = osp.join(root, audio_fpath.replace(".flac", ".txt"))
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
            # copy the audio file
            audio_src = osp.join(root, audio_fpath)
            tgt_audio_fname = audio_fpath.split("/")[-1]
            audio_dst = osp.join(output_dir, "audio_samples", f"{idx}_{tgt_audio_fname}")
            shutil.copyfile(audio_src, audio_dst)
            rows.append([idx, tgt_audio_fname, whisper_transcript, small_model_trans_dict[idx]])
        # write to csv
        rows = sorted(rows, key=lambda x: x[0])
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect hallucinations based on the hallucination detection results csv and the original csv")
    parser.add_argument("--non_hallucinated_tsv", type=str, help="The hallucination detection results tsv")
    parser.add_argument("--original_tsv", type=str, help="The original tsv")
    parser.add_argument("--small_model_trans_tsv", type=str)
    parser.add_argument("--output_dir", type=str, help="The output directory")
    parser.add_argument("--num_samples", type=int, default=1000, help="The number of samples to collect")
    parser.add_argument("--seed", type=int, default=0, help="The random seed")
    args = parser.parse_args()
    collect_hallucinations(args.non_hallucinated_tsv, args.original_tsv, args.small_model_trans_tsv, args.output_dir, args.num_samples, args.seed)