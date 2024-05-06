import os
import os.path as osp
import soundfile as sf
import argparse
from tqdm import tqdm
import re

SAMPLE_RATE = 16000

def time_str_to_frame(time_str, sample_rate=16000):
    h, m, s = map(float, time_str.replace(',', '.').split(':'))
    return int((h * 3600 + m * 60 + s) * sample_rate), time_str.split(',')[-1]

def get_segments_by_srt(srt_file, sample_rate=16000):
    segments = []
    ms_set = set()
    with open(srt_file, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            items = list(map(lambda l: l.strip(), lines[i:i+4]))
            if len(items) < 4:
                continue
            _, timestamp, trans, _ = items
            time_reprs = re.findall(r'\d{2}:\d{2}:\d{2},\d{3}', timestamp)
            start_time = time_reprs[0]
            end_time = time_reprs[1]
            start_frame, ms = time_str_to_frame(start_time, sample_rate)
            end_frame, ms = time_str_to_frame(end_time, sample_rate)
            ms_set.add(ms)
            segments.append((start_frame, end_frame, trans))
    
    return segments, ms_set

def segment_audio(audio_dir, srt_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of audio files in the audio directory
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3') or f.endswith('.wav') or f.endswith('.flac')]
    # audio_files = audio_files[:1] # for testing
    unsuccesful_logfw = open(osp.join(audio_dir, '../unsuccesful.log'), 'w')
    # Iterate over each audio file
    for audio_file in tqdm(audio_files, total=len(audio_files)):
        tqdm.write(f"Processing {audio_file}")
        # Construct the input and output file paths
        audio_postfix = '.' + audio_file.split('.')[-1]
        input_file = osp.join(audio_dir, audio_file)
        srt_file = osp.join(srt_dir, audio_file.replace(audio_postfix, '.srt'))
        output_subdir = osp.join(output_dir, audio_file.replace(audio_postfix, ''))
        segments, ms_set = get_segments_by_srt(srt_file, sample_rate=SAMPLE_RATE)
        if len(ms_set) < 5:
            tqdm.write(f"ms_set: {ms_set} too small, skip {audio_file}")
            print(f"ms_set: {ms_set} too small, skip {audio_file}", file=unsuccesful_logfw, flush=True)
            continue
        if not osp.exists(output_subdir):
            os.makedirs(output_subdir, exist_ok=True)
        # Load the audio file
        audio, sr = sf.read(input_file)
        length = len(audio)
        assert sr == SAMPLE_RATE, f"Sample rate mismatch: {sr} != {SAMPLE_RATE}"
        # perform segmentation
        for i, (start_frame, end_frame, trans) in enumerate(segments):
            if end_frame > length:
                tqdm.write(f"end_frame {end_frame} > length {length}")
                continue
            segment_audio = audio[start_frame:end_frame]
            output_file = osp.join(output_subdir, f"{i:05d}.flac")
            sf.write(output_file, segment_audio, SAMPLE_RATE)
            with open(osp.join(output_subdir, f"{i:05d}.txt"), 'w') as fw:
                print(trans, file=fw, flush=True)

def main(args):
    print(args)
    audio_dir = args.audio_dir
    srt_dir = args.srt_dir
    output_dir = args.output_dir
    segment_audio(audio_dir, srt_dir, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_dir",
        default="",
        help="directory containing audio files",
    )
    parser.add_argument(
        "--srt_dir",
        default="",
        help="directory containing srt files",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="directory to save the segmented audio files",
    )
    args = parser.parse_args()

    main(args)