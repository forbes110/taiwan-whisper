import os
import os.path as osp
import argparse
import glob
import numpy as np
from tqdm import tqdm
import soundfile as sf
from collections import defaultdict
from utils.transcript_readers import read_vtt, timecode_to_seconds

def check_hallucination(text, **kwargs):
    def _length_checker(text, threshold=150):
        if len(text) < threshold:
            return False
        return True

    def _char_ngram_checker(text, n=5, threshold=5):
        ngram_counts = defaultdict(lambda: 0)
        if len(text) < n:
            return False
        for i in range(len(text) - n + 1):
            ngram = text[i:i+n]
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

def main(args):
    print(args)
    print(f"Transcript directory: {args.trans_dir}")
    trans_fpaths = glob.glob(osp.join(args.trans_dir, "*.vtt"))
    trans_fpaths = sorted(filter(lambda x: "en" not in x, trans_fpaths))
    print(f"Total fpaths: {len(trans_fpaths)}, e.g.: {trans_fpaths[:5]}")
    trans_fpaths_search_spaces = trans_fpaths
    hallucinated_candidates = []
    total_segments = 0
    for trans_fpath in tqdm(trans_fpaths_search_spaces, total=len(trans_fpaths_search_spaces)):
        # print(f"Checking {trans_fpath}...")
        segments = read_vtt(trans_fpath)
        for s, e, t in segments[:5]:
            is_hallucinated = check_hallucination(t)
            if is_hallucinated:
                hallucinated_candidates.append((osp.basename(trans_fpath), s, e, t))
        total_segments += len(segments)
    for n, s, e, t in hallucinated_candidates:
        print(f"Name: {n}, Start: {s}, End: {e}, Text: {t}")
        print("")
    print(f"Total segments: {total_segments}")
    print(f"Total hallucinated segments: {len(hallucinated_candidates)}")
    # ratio
    print(f"Hallucination ratio: {len(hallucinated_candidates) / total_segments}")

    # get the hallucinated segments
    audio_dir = args.trans_dir.replace('trans', 'raw')
    output_dir = args.trans_dir.replace('trans', 'hallucinated')
    os.makedirs(output_dir, exist_ok=True)
    for n, s, e, t in tqdm(hallucinated_candidates, total=len(hallucinated_candidates), desc="Extracting hallucinated segments"):
        audio_fpath = osp.join(audio_dir, n.replace('.vtt', '.flac'))
        if not osp.exists(audio_fpath):
            print(f"Audio file not found: {audio_fpath}")
            continue
        data, sr = sf.read(audio_fpath)
        start_frame = int(timecode_to_seconds(s) * sr)
        end_frame = int(timecode_to_seconds(e) * sr)
        assert end_frame > start_frame, f"End frame {end_frame} <= Start frame {start_frame} @ {audio_fpath}"
        audio_segment = data[start_frame:end_frame]
        base_name = osp.basename(n).split('.')[0]
        output_fname = f"{base_name}_{start_frame}-{end_frame}.flac"
        output_subdir = osp.join(output_dir, base_name)
        if not osp.exists(output_subdir):
            os.makedirs(output_subdir, exist_ok=True)
        output_fpath = osp.join(output_subdir, output_fname)
        sf.write(output_fpath, audio_segment, sr)
        with open(output_fpath.replace('.flac', '.txt'), "w") as f:
            f.write(t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trans_dir",
        default="",
        help="the directory containing the transcripts",
    )
    args = parser.parse_args()

    main(args)