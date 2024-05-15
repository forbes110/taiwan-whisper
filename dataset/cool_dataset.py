import os
import os.path as osp
import argparse
from copy import deepcopy
import datasets
from datasets import IterableDataset, Features
from torch.utils.data import DataLoader, Dataset
import soundfile as sf
import re

sampling_rate = 16000

# class CoolIterableDataset(IterableDataset):
#     def __init__(self, metadata_dir, split, **kwargs):
#         self.metadata_dir = metadata_dir
#         self.split = split
#         self.kwargs = kwargs
#         super().__init__()
def _trim_last_segment(feature: Features):
    timestamp_re_pattern = r"<\|\d{1,2}\.\d{2}\|>"
    timestamps = re.findall(timestamp_re_pattern, feature["whisper_transcript"])
    if len(timestamps) > 1:
        last_timestamp = timestamps[-1]
        # 1. "<|0.00|>...<|29.00|><|29.00|><|continued|>" -> "<|0.00|>...<|29.00|>"
        # 2. "<|0.00|>...<|29.00|>" -> "<|0.00|>...<|29.00|>"
        feature["whisper_transcript"] = feature["whisper_transcript"].split(last_timestamp)[0] + last_timestamp 
        trim_start_frame = int(float(last_timestamp[2:-2]) * sampling_rate)
        if trim_start_frame < len(feature["audio"]["array"]):
            feature["audio"]["array"] = feature["audio"]["array"][:trim_start_frame]
    return feature

def _append_last_segment(feature: Features):
    # re search pattern: <|0.00|> ~ <|30.00|>. re expression: <\|(\d{2}\.\d{2})\|>
    special_tokens_re_pattern = r"<\|[\w\.]{1,12}\|>"
    special_tokens_of_whisper_transcript = re.findall(special_tokens_re_pattern, feature["whisper_transcript"])
    if "<|continued|>" in special_tokens_of_whisper_transcript:
        timestamp_before_continued = special_tokens_of_whisper_transcript[special_tokens_of_whisper_transcript.index("<|continued|>") - 1]
        new_transcript = feature["whisper_transcript"].split(timestamp_before_continued)[0]
        new_transcript += feature["last_segment_transcript"] + "<|endoftext|>"
        feature["whisper_transcript"] = new_transcript
    else:
        new_transcript = feature["whisper_transcript"].split("<|endoftext|>")[0]
        new_transcript += feature["last_segment_transcript"] + "<|endoftext|>"


last_segment_handlers = {
    "trim": _trim_last_segment,
    "append": _append_last_segment
}

def _get_feature_by_audio_fpath(audio_fpath, last_segment_handler_type="trim"):
    feature = Features()
    feature["audio"] = {
        "path": audio_fpath,
        "sampling_rate": sampling_rate,
        "array": sf.read(audio_fpath)[0]
    }
    with open(audio_fpath.replace(".flac", ".txt"), "r") as trans_fr:
        lines = trans_fr.readlines()
        whisper_transcript = lines[0].strip().split("<|endoftext|>")[0] # remove <|endoftext|>
        end_transcript = lines[2].strip()
        prev_transcript = lines[4].strip().split("<|endoftext|>")[0] # remove <|endoftext|>
        feature["whisper_transcript"] = whisper_transcript
        feature["last_segment_transcript"] = end_transcript
        feature["condition_on_prev"] = "<|startofprev|>" + prev_transcript
        if "<|continued|>" in prev_transcript:
            timestamp_re_pattern = r"<\|\d{1,2}\.\d{2}\|>"
            timestamps = re.findall(timestamp_re_pattern, feature["condition_on_prev"])
            if len(timestamps) > 1:
                last_timestamp = timestamps[-1]
                # 1. "<|0.00|>...<|29.00|><|29.00|><|continued|>" -> "<|0.00|>...<|29.00|>"
                # 2. "<|0.00|>...<|29.00|>" -> "<|0.00|>...<|29.00|>"
                feature["condition_on_prev"] = feature["condition_on_prev"].split(last_timestamp)[0] + last_timestamp
                feature["condition_on_prev"].replace("<|continued|>", "") # ensure that there is no "<|continued|>" in the condition_on_prev
        feature = last_segment_handlers[last_segment_handler_type](feature)
    return feature

def cool_data_generator(audio_fpaths, last_segment_handler_type="trim"):
    for audio_fpath in audio_fpaths:
        feature = _get_feature_by_audio_fpath(audio_fpath, last_segment_handler_type=last_segment_handler_type)
        yield feature

def load_cool_dataset(manifest_fpath, root=None) -> IterableDataset:
    print(f"Loading cool dataset from {manifest_fpath}")
    audio_fpaths = []
    with open(manifest_fpath, "r") as fr:
        if root is None:
            root = fr.readline().strip()
        else:
            _ = fr.readline()
        for line in fr:
            audio_fpath = osp.join(root, line.strip())
            audio_fpaths.append(audio_fpath)
    ex_feature = Features()
    ex_feature["audio"] = "dummy"
    ex_feature["text"] = "dummy"
    ex_feature['whisper_transcript'] = "dummy"
    ex_feature['last_segment_transcript'] = "dummy"
    ex_feature['condition_on_prev'] = "dummy"
    # cool_dataset = IterableDataset.from_generator(cool_data_generator, features=ex_feature, gen_kwargs={"audio_fpaths": audio_fpaths, "last_segment_handler_type": "append"})
    cool_dataset = IterableDataset.from_generator(cool_data_generator, features=ex_feature, gen_kwargs={"audio_fpaths": audio_fpaths})

    return cool_dataset, audio_fpaths
        

def main(args):
    print(args)
    ds, audio_fpaths = load_cool_dataset(args.manifest, args.root)
    ds = ds.cast_column(
        "audio",
        datasets.features.Audio(sampling_rate=sampling_rate),
    )
    for i, item in enumerate(ds):
        print(i, item)
        print(item["audio"]["array"].shape)
        if i == 3:
            break
    # print(ds.features)
    # test shuffle 
    # dl = DataLoader(ds, batch_size=2, num_workers=4)
    # for i, batch in enumerate(dl):
    #     print(i, batch)
    #     if i == 10:
    #         break
    # for i, batch in enumerate(dl):
    #     print(i, batch)
    #     if i == 10:
    #         break
    
    # ds = ds.shuffle(seed=42, buffer_size=1)
    # dl = DataLoader(ds, batch_size=2, num_workers=4)
    # for i, batch in enumerate(dl):
    #     print(i, batch)
    #     if i == 10:
    #         break
    # test dataloader
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=None,
        help="a sample arg",
    )
    parser.add_argument(
        "--manifest",
        default="",
        required=True,
        help="a sample arg",
    )
    args = parser.parse_args()

    main(args)