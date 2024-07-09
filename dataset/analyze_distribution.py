from dataset.prepare_dataset import analyze_categories
from collections import defaultdict
import csv

# dir = "/root/distil-whisper/corpus/ntu_cool/data/train/sub_categorized/raw"
# analyze_categories(dir)
tsv_fpath = "/root/distil-whisper/corpus/ntu_cool/data/train/sub_categorized/raw/categories.tsv"
def analyze_by_tsv(tsv_fpath):
    categories = defaultdict(lambda : {
        'count': 0,
        'frames': 0
    })
    with open(tsv_fpath, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            categories[row[0]]['count'] += 1
            categories[row[0]]['frames'] += int(row[2])
    return categories

categories = analyze_by_tsv(tsv_fpath)
total_videos = 0
total_frames = 0
total_seconds = 0
for k, v in categories.items():
    count = v['count']
    frames = v['frames']
    seconds = frames / 16_000
    print(f"{k}: {count} videos, {frames} frames, {seconds} seconds, {seconds/3600} hours")
    total_videos += count
    total_frames += frames
    total_seconds += seconds
print(f"Total: {total_videos} videos, {total_frames} frames, {total_seconds} seconds, {total_seconds/3600} hours")