import os
import csv
import os.path as osp
import argparse

SAMPLE_RATE = 16000
VALID_SID_FIRST_CHARS = [
    '0', # General Education
    '1', # Liberal Arts
    '2', # Science
    '3', # Social Science
    '4', # MED
    '5', # Engineering
    '6', # Bio-resource and Agriculture
    '7', # Management
    '8', # Public Health
    '9', # EECS
    'A', # Law School 10
    'B', # Life Science 11
    'E', # Continuing Education Division 12
    'K', # Advanced Technology 13
    'F', # D-school 14
    'H', # D-school 15
    'Z', # D-school 16
    'P', # Program 17
    'Q', # Academic Writing Center 18
]

def normalize_sid(raw_sid):
    if raw_sid is None:
        return None
    items = raw_sid.split(':')
    sid = items[-1]
    if len(items) == 3:
        sid = items[1]
    return sid

def is_valid_sid(sid):
    if sid is None:
        return False
    if len(sid) == 0:
        return False
    items = sid.split('_')
    if len(items) != 2:
        return False
    if items[0][0] not in VALID_SID_FIRST_CHARS:
        return False
    return True

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


def read_vid_to_other_ids_mapping(vid_to_other_ids_csv_fpath, normalized_sid=True):
    vid_to_other_ids = {}
    with open(vid_to_other_ids_csv_fpath, 'r') as f:
        reader = csv.reader(f)
        _columns = next(reader)
        for row in reader:
            if len(row) != 3:
                print(f"Error: row {row} has {len(row)} columns, expected 3 columns")
                continue
            vid = row[0]
            cid = row[1]
            sid = row[2]
            if normalized_sid:
                sid = normalize_sid(sid)
            vid_to_other_ids[vid] = {'cid': cid, 'sid': sid}
    return vid_to_other_ids

def read_sid_to_course_name_mapping(sid_to_course_name_csv_fpath):
    sid_to_course_name = {}
    with open(sid_to_course_name_csv_fpath, 'r') as f:
        reader = csv.reader(f)
        _columns = next(reader)
        for row in reader:
            if len(row) != 3:
                print(f"Error: row {row} has {len(row)} columns, expected 3 columns")
                continue
            sid = row[0]
            zh_course_name = row[1]
            en_course_name = row[2]
            sid_to_course_name[sid] = {'zh': zh_course_name, 'en': en_course_name}
    return sid_to_course_name

def main(args):
    print(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="",
        help="a sample arg",
    )
    args = parser.parse_args()

    main(args)