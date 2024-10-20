# create transciprtion file readers (.vtt, .srt, .txt)
# additional function: time code to seconds converter

# vtt reader
def read_vtt(vtt_fpath):
    segments = []
    with open(vtt_fpath, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "-->" in line:
                items = line.split("-->")
                if len(items) != 2:
                    continue
                start, end = items
                start = start.strip()
                end = end.strip()
                text = lines[i + 1].strip()
                segments.append((start, end, text))
    return segments

# time code to seconds converter
def timecode_to_seconds(timecode):
    timecode = timecode.strip()
    timecode_items = timecode.split(":")
    seconds = float(timecode_items[-1])
    if len(timecode_items) >= 2:
        seconds += int(timecode_items[-2]) * 60
    if len(timecode_items) >= 3:
        seconds += int(timecode_items[-3]) * 3600
    return seconds