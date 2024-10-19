import os
import numpy as np
import os.path as osp
import argparse
import datasets
from copy import deepcopy
from time import sleep
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import IterableDataset, Features
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from utils.hallucination_detector import WhisperSmallModelDetector
import soundfile as sf
import re

sampling_rate = 16000
logger = get_logger(__name__)

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
    audio_data = sf.read(audio_fpath)[0]
    feature["audio"] = {
        "path": audio_fpath,
        "sampling_rate": sampling_rate,
        "array": audio_data
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

def load_audio_fpaths(manifest_fpath, root=None):
    audio_fpaths = []
    with open(manifest_fpath, "r") as fr:
        if root is None:
            root = fr.readline().strip()
        else:
            _ = fr.readline()
        for line in fr:
            audio_fpath = osp.join(root, line.strip())
            audio_fpaths.append(audio_fpath)
    return audio_fpaths

def load_cool_dataset(manifest_fpath, root=None) -> IterableDataset:
    print(f"Loading cool dataset from {manifest_fpath}")
    audio_fpaths = load_audio_fpaths(manifest_fpath, root=root)
    ex_feature = Features()
    ex_feature["audio"] = "dummy"
    ex_feature["text"] = "dummy"
    ex_feature['whisper_transcript'] = "dummy"
    ex_feature['last_segment_transcript'] = "dummy"
    ex_feature['condition_on_prev'] = "dummy"
    # cool_dataset = IterableDataset.from_generator(cool_data_generator, features=ex_feature, gen_kwargs={"audio_fpaths": audio_fpaths, "last_segment_handler_type": "append"})
    cool_dataset = IterableDataset.from_generator(cool_data_generator, features=ex_feature, gen_kwargs={"audio_fpaths": audio_fpaths})

    return cool_dataset, audio_fpaths

# I construct this dataset for using accelerate to perform pre-filtering on the training set
class PreFilterCoolASRDataset(Dataset):
    def __init__(self, metadata_fpath, root=None, **kwargs):
        self.metadata_fpath = metadata_fpath
        logger.info(f"Loading cool dataset from {metadata_fpath}")
        self.kwargs = kwargs
        super().__init__()
        # parse_metadata
        self.audio_fpaths = load_audio_fpaths(metadata_fpath, root=root)

    def shuffle(self, seed=42):
        pass

    def set_epoch(self, epoch):
        pass

    def __getitem__(self, idx):
        audio_fpath = self.audio_fpaths[idx]
        logger.info(f"Loading audio from {audio_fpath}")
        audio_data, sr = sf.read(audio_fpath)
        assert sr == sampling_rate, f"Sampling rate {sr} is not 16kHz"
        # if audio data is not equal to 30 secs, trim it or append it 
        if len(audio_data) != 30 * sr:
            if len(audio_data) < 30 * sr:
                audio_data = np.pad(audio_data, (0, 30 * sr - len(audio_data)))
            else:
                audio_data = audio_data[:30 * sr]
        feature = {'idx': idx, 'path': audio_fpath, 'array': audio_data}
        # feature = {'idx': idx, 'path': audio_fpath}
        # feature = _get_feature_by_audio_fpath(audio_fpath)
        # feature['idx'] = idx
        return feature # for testing accelerate behavior
        # feature = _get_feature_by_audio_fpath(audio_fpath)
        # return feature
    
    def __len__(self):
        return len(self.audio_fpaths)


def test_load_cool_dataset(manifest, root):        
    ds, audio_fpaths = load_cool_dataset(manifest, root)
    ds = ds.cast_column(
        "audio",
        datasets.features.Audio(sampling_rate=sampling_rate),
    )
    for i, item in enumerate(ds):
        print(i, item)
        print(item["audio"]["array"].shape)
        if i == 3:
            break
    print(ds.features)
    # test shuffle 
    dl = DataLoader(ds, batch_size=2, num_workers=4)
    for i, batch in enumerate(dl):
        print(i, batch)
        if i == 10:
            break
    for i, batch in enumerate(dl):
        print(i, batch)
        if i == 10:
            break

def main(args):
    
    print(args)
    # test load_cool_dataset
    # test_load_cool_dataset(args.manifest, args.root)
    # test self-defined dataset with accelerate
    accelerator = None
    accelerator = Accelerator(

    )
    
    dataset = PreFilterCoolASRDataset(args.manifest)
    if accelerator.is_main_process:
        print(f"Total samples: {len(dataset)}")
        
    # distributed_sampler = DistributedSampler(dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index)
    hallucination_dectector = WhisperSmallModelDetector(small_model_card='openai/whisper-base', accelerator=accelerator)
    dataloader_for_test = DataLoader(dataset, 
        batch_size=64, 
        num_workers=16,
        shuffle=False, 
        collate_fn=hallucination_dectector.collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    
    print(f"Prepare dataloader for testing ...")
    dataloader_for_test = accelerator.prepare(dataloader_for_test)
    print(f"Prepare dataloader for testing ... Done")
    
    if accelerator.is_main_process:
        print(f"Dataloader size: {len(dataloader_for_test)}")
    steps_inference_progress_bar = tqdm(
        range(len(dataloader_for_test)), desc="Inference segments ... ", position=0, disable=not accelerator.is_local_main_process
    )
    
    # get rank by os
    rank = int(os.environ['RANK'])
    print(f"Rank: {rank}")
    # overall_idxs = []
    # overall_preds = []
    
    accelerator.wait_for_everyone()
    with open(f"./inference_results/whisper_base/idx_hyp.{rank}.txt", "w") as fw:
        for i, batch in enumerate(dataloader_for_test):
            # logger.info(i, batch['idx'], batch['path'], batch['array'].shape)
            # print(i, batch['idx'], batch['path'], batch['array'].shape)
            # print(i, batch.keys(), batch['input_features'].shape, batch['idx'])
            idxs, preds_str = hallucination_dectector.generate(batch)
            # assert len(idxs) == len(preds), f"Length of idxs and pred_str should be the same, current idxs: {len(idxs)}, pred_str: {len(preds)}"
            assert len(idxs) == len(preds_str), f"Length of idxs and pred_str should be the same, current idxs: {len(idxs)}, pred_str: {len(preds_str)}"
            steps_inference_progress_bar.update(1)
            # sleep(1.0 * np.random.random()+ 1.0)
            # gather all results
            for idx, pred_str in zip(idxs, preds_str):
                fw.write(f"{idx}\t{pred_str}\n\n")
                fw.flush()
            # if i % 100 == 0:
            #     accelerator.wait_for_everyone()
            # if i > 10:
            #     break
    # sort
    # overall_idxs = np.concatenate(overall_idxs)
    # overall_preds = np.concatenate(overall_preds)
    # if accelerator.is_main_process:
    #     for idx, pred in zip(overall_idxs, overall_preds):
    #         audio_fpath = dataset.audio_fpaths[idx]
    #         # print("-------------------------------------")
    #         print(f"idx: {idx}")
            # print(f"Audio fpath: {audio_fpath}")
            # print(f"Pred: {pred_str}")

    
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