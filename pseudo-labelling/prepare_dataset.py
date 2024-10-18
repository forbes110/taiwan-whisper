import os
import os.path as osp
import argparse
import glob
import logging
import time
import gc
import soundfile as sf
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict
from functools import partial
import csv

SAMPLE_RATE = 16000
SEGMENT_LENGTH = 30 * SAMPLE_RATE  # (secs * sample_rate = frames)
ADD_CONTINUED_TOKEN_THRESHOLD = 1.0  # (secs)

"""
# TODO: 
1. how to handle overlapping?
2. why there are txt after 30s in 30?, 
    e.g., 
    <|18.40|> ... 現在最流行的直接你什麼是你沒做到這個打你媽確診 想要超來講你媽確診<|30.00|>

    0,35.64s,36.80s,現在最流行的是直接
    0,36.80s,37.86s,什麼事情沒做好
    0,37.86s,38.20s,直接打
    0,38.20s,39.04s,你媽確診
    0,41.10s,42.64s,反正每個都自己噴你
    0,42.64s,43.20s,你媽確診
"""

def frame_diff_to_timestamp(frame_diff, sample_rate=SAMPLE_RATE):
    residual = frame_diff % 320
    if 320 - residual > 5 and residual > 5:
        print(f"Warning: frame_diff {frame_diff} is not very close to a multiple of 320")
        # round frame_diff to the nearest 320 frames
        frame_diff = round(frame_diff / 320) * 320
    sec_diff = frame_diff / sample_rate # if frame_diff is a multiple of 320, then sec_diff is with resolution of 0.02s
    # use max min function to ensure sec_diff is within [0.00, 30.00]
    sec_diff = max(0.00, min(30.00, sec_diff))
    # return token format <|sec_diff:.2f|>
    return f"<|{sec_diff:.2f}|>"

def timecode_to_seconds(timecode):
    timecode = timecode.strip()
    timecode_items = timecode.split(":")
    seconds = float(timecode_items[-1])
    if len(timecode_items) >= 2:
        seconds += int(timecode_items[-2]) * 60
    if len(timecode_items) >= 3:
        seconds += int(timecode_items[-3]) * 3600
    return seconds

def read_pseudo_labels(csv_fpath):
    """
    pseudo_label takes the form
    start, end, text
    0.252, 18.391, Hello
    """
    segments = []
    with open(csv_fpath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  
        for row in reader:
            if len(row) != 3:
                continue
            start, end, text = row
            segments.append((start.strip(), end.strip(), text.strip()))
    return segments

# Segment audio based on transcriptions
def segment_audio_by_trans(audio_trans_pair, segment_output_dir):
    """
    Segment audio based on transcription.

    Args:
        audio_trans_pair (tuple): (audio_file_path, transcription_file_path)
        segment_output_dir (str): output directory for segmented audio

    Returns:
        str: "Success" on success, otherwise (audio_trans_pair, Exception)
    """
    try:
        audio_fpath, trans_fpath = audio_trans_pair
        
        print(f"Segmenting {audio_fpath} based on {trans_fpath}")   
        
        file_name = osp.basename(audio_fpath).split('.')[0]
        
        audio_output_dir = osp.join(segment_output_dir, file_name)
        
        os.makedirs(audio_output_dir, exist_ok=True)
        
        audio_data, sr = sf.read(audio_fpath)
        segments = read_pseudo_labels(trans_fpath)
        prev_end_frame = 0
        prev_text = ""
        cur_text = ""

        for i, segment in enumerate(segments):
            start, end, text = segment
            start_time_in_seconds = timecode_to_seconds(start)
            end_time_in_seconds = timecode_to_seconds(end)
            s_frame = int(start_time_in_seconds * SAMPLE_RATE)
            e_frame = int(end_time_in_seconds * SAMPLE_RATE)

            s_timestamp = frame_diff_to_timestamp(s_frame - prev_end_frame)
            e_timestamp = frame_diff_to_timestamp(e_frame - prev_end_frame)

            if e_frame - prev_end_frame > SEGMENT_LENGTH:
                cur_end_frame = prev_end_frame + SEGMENT_LENGTH
                segmented_audio = audio_data[prev_end_frame:cur_end_frame]
                
                # if cur_end_frame - s_frame > ADD_CONTINUED_TOKEN_THRESHOLD * SAMPLE_RATE:
                #     cur_text += s_timestamp + "<|continued|>"
                cur_text += "<|endoftext|>"
                
                segment_output_fpath = osp.join(audio_output_dir, f"{file_name}_{prev_end_frame}-{cur_end_frame}.flac")
                sf.write(segment_output_fpath, segmented_audio, SAMPLE_RATE)

                with open(osp.join(audio_output_dir, f"{file_name}_{prev_end_frame}-{cur_end_frame}.txt"), 'w') as f:
                    f.write(cur_text + "\n")
                    f.write(s_timestamp + text + e_timestamp + "\n")
                    f.write(prev_text + "\n")

                prev_end_frame = s_frame
                prev_text = cur_text
                cur_text = s_timestamp + text + e_timestamp
            else:
                cur_text += s_timestamp + text + e_timestamp

        return "Success"
    except Exception as e:
        return (audio_trans_pair, e)

# Process all audio files based on transcriptions
def segment_audio(audio_dir, trans_dir, segment_output_dir):
    
    # Build a mapping from video IDs to transcription files
    pseudo_label_fpath = {}
    
    # all pseudo label path generated by whisper-large
    trans_fpaths = list(glob.glob(osp.join(trans_dir, '*.csv'), recursive=True))

    for trans_fpath in tqdm(trans_fpaths, desc="Parsing transcriptions..."):
        file_name = osp.basename(trans_fpath).split('.')[0]
        pseudo_label_fpath[file_name] = trans_fpath
        
    # Find and pair all audio files with their transcriptions
    audio_trans_pairs = []
    audio_fpaths = glob.glob(osp.join(audio_dir, '*.flac'))

    for audio_fpath in audio_fpaths:
        file_name = osp.basename(audio_fpath).split('.')[0]
        trans_fpath = pseudo_label_fpath.get(file_name)
        
        if trans_fpath is None:
            print(f"Warning: No transcription found for {file_name}")
            continue
        
        # audio path, pseudo_label path pair
        audio_trans_pairs.append((audio_fpath, trans_fpath))

    # Process audio files in parallel using multiprocessing
    chunk_size = 100 if len(audio_trans_pairs) > 100 else len(audio_trans_pairs)
    
    
    segment_func = partial(segment_audio_by_trans, segment_output_dir=segment_output_dir)
    
    for i in range(0, len(audio_trans_pairs), chunk_size):
        chunk = audio_trans_pairs[i:i + chunk_size]
        print(f"Processing chunk {i}-{i + len(chunk)} with {args.nprocs} processes...")

        with mp.Pool(processes=args.nprocs) as pool:
            for result in tqdm(pool.imap_unordered(segment_func, chunk), total=len(chunk), desc="Segmenting audio..."):
                if result != "Success":
                    print(f"Error: Failed to segment {result[0]}, error={result[1]}", flush=True)
        gc.collect()
    print("Done")

def main(args):
    print(args)
    segment_audio(args.audio_dir, args.trans_dir, args.segment_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trans_dir", required=True, help="Root directory of transcriptions")
    parser.add_argument("--segment_output_dir", required=True, help="segment output directory")
    parser.add_argument("--audio_dir", required=True, help="audio directory")
    parser.add_argument("--nprocs", type=int, default=8, help="Number of processes for parallel segmentation")
    args = parser.parse_args()

    main(args)
