#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
import glob
import os
import random
import tqdm
import soundfile as sf
import multiprocessing as mp

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "--valid-percent",
        default=0.0,
        type=float,
        metavar="D",
        help="percentage of data to use as validation set (between 0 and 1)",
    )
    parser.add_argument(
        "--output-fname", default="train", type=str, metavar="NAME", help="output fname"
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--ext", default="flac", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    parser.add_argument(
        "--path-must-contain",
        default=None,
        type=str,
        metavar="FRAG",
        help="if set, path must contain this substring for a file to be included in the manifest",
    )
    parser.add_argument(
        "--sort",
        default=False,
        action="store_true",
        help="sort the list of files before writing them to the manifest",
    )
    parser.add_argument(
        "--get-frames",
        default=False,
        action="store_true",
        help="get the number of frames in the audio file",
    )
    return parser

def get_frames(file_path):
    
    frames = sf.info(file_path).frames
    return file_path, frames

def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 1.0

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, "**/*." + args.ext)
    rand = random.Random(args.seed)

    output_fname = args.output_fname
    valid_fname = "valid" if output_fname == "train" else f"{output_fname}-valid"
    valid_f = (
        open(os.path.join(args.dest, f"{valid_fname}.tsv"), "w")
        if args.valid_percent > 0
        else None
    )

    with open(os.path.join(args.dest, f"{output_fname}.tsv"), "w") as train_f:
        print(dir_path, file=train_f)

        if valid_f is not None:
            print(dir_path, file=valid_f)
        
        
        file_paths = [os.path.realpath(fname) for fname in glob.iglob(search_path, recursive=True)]
        if args.get_frames:
            pool = mp.Pool(processes=8)
            file_paths_and_frames = list(tqdm.tqdm(pool.imap_unordered(get_frames, file_paths), total=len(file_paths)))
        else:
            file_paths_and_frames = [(file_path, None) for file_path in file_paths]
        if args.sort:
            file_paths_and_frames = sorted(file_paths_and_frames, key=lambda x: x[0])

        for file_path, frame in file_paths_and_frames:
            if args.path_must_contain and args.path_must_contain not in file_path:
                continue

            dest = train_f if rand.random() > args.valid_percent else valid_f
            if frame is None:
                print(os.path.relpath(file_path, dir_path), file=dest)
            else:
                print(
                    "{}\t{}".format(os.path.relpath(file_path, dir_path), frame), file=dest
                )
    if valid_f is not None:
        valid_f.close()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
