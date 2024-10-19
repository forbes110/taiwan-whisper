# prepare data (categorization, segmentation, timestamp tokenization, metadata collection, etc.) for training

import os
import os.path as osp
import argparse
import glob
import logging
import time
import random
import gc
import soundfile as sf
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict
from dataset.data_utils import read_vid_to_other_ids_mapping, read_sid_to_course_name_mapping, frame_diff_to_timestamp, is_valid_sid, VALID_SID_FIRST_CHARS, SAMPLE_RATE
from utils.transcript_readers import read_vtt, timecode_to_seconds

CATEGORIES = [
    f"{c}00" for c in VALID_SID_FIRST_CHARS
] + ['unknown']

SEGMENT_LENGTH = 30 * SAMPLE_RATE # (secs * sample_rate = frames)
ADD_CONTINUED_TOKEN_THRESHOLD = 1.0 # (secs)

def categorize_audio(audio_dir, output_dir, metadata_dir):
    vid_to_other_ids = read_vid_to_other_ids_mapping(osp.join(metadata_dir, 'vid_cid_sid.csv'), normalized_sid=True)
    sid_to_course_name = read_sid_to_course_name_mapping(osp.join(metadata_dir, 'sid_course_name.csv'))
    def _get_sid_and_course_name(vid):
        other_ids = vid_to_other_ids.get(vid, None)
        if other_ids is None:
            return None
        sid = other_ids.get('sid', None)
        return {'sid': sid, 'course_name': sid_to_course_name.get(sid, None)}
    audio_fpaths = glob.glob(osp.join(audio_dir, '**', '*.flac'), recursive=True)
    print(f"Found {len(audio_fpaths)} audio files")
    audio_fpaths = sorted(audio_fpaths)
    # make subdirectories
    for valid_sid_first_char in VALID_SID_FIRST_CHARS:
        os.makedirs(osp.join(output_dir, valid_sid_first_char + '00'), exist_ok=True)
    os.makedirs(osp.join(output_dir, 'unknown'), exist_ok=True)
    # mv audio files to its corresponding subdirectory
    for i, audio_fpath in tqdm(enumerate(audio_fpaths), total=len(audio_fpaths), desc="Categorizing audio files"):
        vid = osp.basename(audio_fpath).split('.')[0]
        sid_and_course_name = _get_sid_and_course_name(vid)
        category = 'unknown'
        if sid_and_course_name is None:
            print(f"Warning: no sid and course name found for vid {vid}")
        else:
            sid = sid_and_course_name['sid']
            course_name = sid_and_course_name['course_name']
            if is_valid_sid(sid):
                category = sid[0] + '00'
            else:
                print(f"Warning: invalid sid {sid} for vid {vid}, categorize to 'unknown'")
        
        target_fpath = osp.join(output_dir, category, vid + '.flac')
        os.rename(audio_fpath, target_fpath)
        print(f"moved {audio_fpath} to {target_fpath}.")

def analyze_categories(output_dir):
    time_distribution_over_categories = defaultdict(int)
    categories = CATEGORIES
    with open(osp.join(output_dir, 'categories.tsv'), 'w') as fw:
        for category in tqdm(categories, total=len(categories), desc="Analyzing categories"):
            audio_fpaths = glob.glob(osp.join(output_dir, category, '*.flac'))
            frames = 0
            for audio_fpath in tqdm(audio_fpaths, total=len(audio_fpaths), desc=f"Analyzing category {category}"):
                finfo = sf.info(audio_fpath)
                assert finfo.samplerate==SAMPLE_RATE, f"Error: {audio_fpath} has sample rate {finfo.samplerate}, expected {SAMPLE_RATE}"
                frames += finfo.frames
                fw.write(f"{category}\t{audio_fpath}\t{finfo.frames}\n")
            time_distribution_over_categories[category] = frames / SAMPLE_RATE
    print("Time distribution over categories:")
    for category, time in time_distribution_over_categories.items():
        print(f"{category}: {time} seconds ({time/3600} hours)")

# segment audio based on transcriptions
def segment_audio_by_trans(audio_trans_pair):
    try:
        audio_fpath, trans_fpath = audio_trans_pair
        print(f"segmenting {audio_fpath} based on {trans_fpath}")
        vid = osp.basename(audio_fpath).split('.')[0]
        audio_output_dir = osp.join(osp.dirname(audio_fpath), vid)
        # if osp.exists(audio_output_dir):
        #     print(f"Warning: {audio_output_dir} already exists, skip")
        #     return "Success"
        os.makedirs(audio_output_dir, exist_ok=True)
        # data, sr = sf.read(audio_fpath)
        segments = read_vtt(trans_fpath)
        prev_end_frame = 0
        prev_text = ""
        cur_text = ""
        # segmented_data_with_trans_list = []
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
                # segmented_data = data[prev_end_frame:cur_end_frame]
                if cur_end_frame - s_frame > ADD_CONTINUED_TOKEN_THRESHOLD * SAMPLE_RATE:
                    cur_text += s_timestamp
                    cur_text += "<|continued|>"
                cur_text += "<|endoftext|>"
                # segmented_data_with_trans_list.append((segmented_data, cur_text))
                # segment_output_fpath = osp.join(audio_output_dir, f"{vid}_{prev_end_frame}-{cur_end_frame}.flac")
                # sf.write(segment_output_fpath, segmented_data, SAMPLE_RATE)
                with open(osp.join(audio_output_dir, f"{vid}_{prev_end_frame}-{cur_end_frame}.txt"), 'w') as f:
                    f.write(cur_text + "\n")
                    f.write("\n" + s_timestamp + text + e_timestamp + "\n")
                    f.write("\n" + prev_text + "\n")
                prev_end_frame = s_frame
                prev_text = cur_text
                s_timestamp = frame_diff_to_timestamp(s_frame - prev_end_frame)
                e_timestamp = frame_diff_to_timestamp(e_frame - prev_end_frame)
                cur_text = s_timestamp + text + e_timestamp
            else:
                cur_text += s_timestamp
                cur_text += text
                cur_text += e_timestamp
        return "Success"
    except Exception as e:
        return (audio_trans_pair, e)
# perform reversable audio segmentation based on transcriptions
def segment_audio(output_dir, trans_dir):
    # pairing audio fpath and trans fpath
    vid_to_trans_fpath = {}
    trans_fpaths = glob.glob(osp.join(trans_dir, '*.vtt'), recursive=True)
    trans_fpaths = list(filter(lambda x: not 'en' in x, trans_fpaths))
    for trans_fpath in tqdm(trans_fpaths, total=len(trans_fpaths), desc=f"Parsing trans fpaths..."):
        vid = osp.basename(trans_fpath).split('.')[0]
        vid_to_trans_fpath[vid] = trans_fpath
    # geting categorized and sorted audio fpaths
    audio_fpaths_by_category = defaultdict(list)
    categories = CATEGORIES
    for category in categories:
        audio_fpaths_in_category = glob.glob(osp.join(output_dir, category, '*.flac'))
        for audio_fpath in audio_fpaths_in_category:
            vid = osp.basename(audio_fpath).split('.')[0]
            trans_fpath = vid_to_trans_fpath.get(vid, None)
            if trans_fpath == None:
                print(f"Warning: no transcription found for vid {vid}")
                continue
            audio_fpaths_by_category[category].append((audio_fpath, trans_fpath))
        audio_fpaths_by_category[category] = sorted(audio_fpaths_by_category[category], key=lambda x: x[0])
    # segment audio based on transcriptions
    
    chunk_size = 100
    target_category_index = args.category_index
    if target_category_index != -1:
        categories = [categories[target_category_index]]
    for category in categories:
        audio_trans_pairs_in_category = audio_fpaths_by_category[category]
        print(f"Processing category {category} with {len(audio_trans_pairs_in_category)} valid audio-transcript pairs")
        audio_trans_pairs_in_category = audio_trans_pairs_in_category
        # for audio_trans_pair in tqdm(audio_trans_pairs_in_category, total=len(audio_trans_pairs_in_category), desc=f"Segmenting category {category}..."):
        #     result = segment_audio_by_trans(audio_trans_pair)
        #     if result != "Success":
        #         print(f"Error: failed to segment audio (audio_fpath, trans_fpath)={audio_trans_pair}, error={result}")
        # process by chunks
        for i in range(0, len(audio_trans_pairs_in_category), chunk_size):
            end_i = min(i + chunk_size, len(audio_trans_pairs_in_category))
            audio_trans_pairs_in_category_chunk = audio_trans_pairs_in_category[i:end_i]
            print(f"Using multiprocessing, number of processes={args.nprocs}")
            with mp.Pool(processes=args.nprocs) as pool:
                for result in tqdm(pool.imap_unordered(segment_audio_by_trans, audio_trans_pairs_in_category_chunk), total=len(audio_trans_pairs_in_category_chunk), desc=f"Segmenting category {category}, index={i}-{end_i}..."):
                    if result != "Success":
                        print(f"Error: failed to segment audio (audio_fpath, trans_fpath)={result[0]}, error={result[1]}", flush=True)
            print(f"Processed chunk {i}-{end_i}!", flush=True)
        gc.collect()
    print("Done")
    

def main(args):
    print(args)
    audio_dir = args.audio_dir
    trans_dir = args.trans_dir
    output_dir = args.output_dir
    metadata_dir = args.metadata_dir
    # categorize_audio(audio_dir, output_dir, metadata_dir)
    # analyze_categories(output_dir)
    segment_audio(output_dir, trans_dir) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_dir",
        required=True,
        help="root directory of audio files",
    )
    parser.add_argument(
        "--trans_dir",
        required=True,
        help="root directory of transcriptions",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="output directory",
    )
    parser.add_argument(
        "--metadata_dir",
        required=True,
        help="metadata directory",
    )
    parser.add_argument(
        "--nprocs", 
        type=int,
        default=8,
        help="number of processes for segmenting audio files",
    )
    parser.add_argument(
        "--category_index",
        type=int,
        default=-1,
        help="category index to process (default: -1, process all categories)",
    )

    args = parser.parse_args()

    main(args)