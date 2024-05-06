import os
import os.path as osp
import argparse
import tqdm

SAMPLE_RATE = 16000

def main(args):
    print(args)
    with open(args.input, 'r') as fr:
        lines = fr.readlines()
    total_frames = 0
    total_seconds = 0.
    max_frames = (0, 0)
    for i, l in tqdm.tqdm(enumerate(lines[1:]), total=len(lines[1:])):
        vid, frames_str = l.strip().split('\t')
        # print(frames_str)
        frames = int(frames_str)
        if frames > max_frames[1]:
            max_frames = (i, frames)
        total_frames += frames
        total_seconds += frames / SAMPLE_RATE
    total_lines = len(lines[1:])
    print(f"Total frames: {total_frames}")
    print(f"Total audios: {total_lines}")
    print("==================================")
    print(f"Average frames: {total_frames / total_lines}")
    print(f"Max frames: {max_frames}")
    print(f"Total seconds: {total_seconds}")
    print(f"{total_seconds // 3600} hours, {total_seconds % 3600 // 60} minutes and {total_seconds % 60} seconds")
    print(f"Average seconds: {total_seconds / total_lines}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="",
        help="a sample arg",
    )
    args = parser.parse_args()

    main(args)