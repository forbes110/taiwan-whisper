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
import torch
import soundfile as sf
import re
from transformers import (
    AddedToken,
    WhisperForConditionalGeneration, 
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
)

sampling_rate = 16000
logger = get_logger(__name__)

class WhisperSmallModelDetector(object):
    def __init__(self, small_model_card='openai/whisper-base', accelerator=None):
        self.model = WhisperForConditionalGeneration.from_pretrained(small_model_card, low_cpu_mem_usage=True)
        self.processor = WhisperProcessor.from_pretrained(small_model_card)
        self.tokenizer = WhisperTokenizerFast.from_pretrained(small_model_card, use_fast=True)
        self.accelerator = accelerator
        
        # override timestamp tokens until tokenizer issues are fixed in transformers --> what issue? 
        timestamps = [AddedToken("<|%.2f|>" % (i * 0.02), lstrip=False, rstrip=False) for i in range(1500 + 1)]
        
        # with logging.disable(logging.WARNING):
        self.tokenizer.add_tokens(timestamps)
        self.tokenizer.set_prefix_tokens(language='zh', task='transcribe', predict_timestamps=True)
        self.gen_kwargs = {
            "max_length": 448,
            "num_beams": 1,
            "return_timestamps": True,
            "language": 'zh',
            "task": 'transcribe',
        }
        # other kwargs
        self.input_padding = "longest"
        # prepare model
        self.model.eval()
        if self.accelerator is not None:
            self.model = self.model.to(self.accelerator.device)
    
    def collate_fn(self, features):
        # batch keys: 'idx', 'path', 'array'
        inputs = self.processor.feature_extractor(
            [feature['array'] for feature in features],
            sampling_rate=16000,
        )
        # print(inputs.input_features.shape)
        input_features = {
            'input_features': inputs.input_features
        }
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.input_padding,
            return_tensors="pt",
        )
        batch['idx'] = torch.from_numpy(np.array([feature['idx'] for feature in features])).long().to(batch['input_features'].device)
        # batch['additional'] = "hello"
        # print(batch)
        # assert False
        return batch # dict of tensors

    def generate(self, batch):
        with torch.no_grad():
            output_ids = self.model.generate(batch["input_features"], **self.gen_kwargs)
            # output_ids = self.accelerator.pad_across_processes(output_ids, dim=1, pad_index=self.tokenizer.pad_token_id)
            # pad to longest generated sequence
            # output_ids = torch.nn.functional.pad(output_ids, (0, 448 - output_ids.shape[1]), value=self.tokenizer.pad_token_id)
        # print(output_ids.shape)
        preds = output_ids
        preds_str = self.tokenizer.batch_decode(preds, skip_special_tokens=True, decode_with_timestamps=False)
        
        # return [(idx.item(), pred_s) for idx, pred_s in zip(batch['idx'], pred_str)]
        return batch['idx'].cpu().numpy(), preds_str
            

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

# I construct this dataset for using accelerate to perform pre-filtering on the training set
class PreFilterASRDataset(Dataset):
    def __init__(self, metadata_fpath, root=None, **kwargs):
        self.metadata_fpath = metadata_fpath
        logger.info(f"Loading cool dataset from {metadata_fpath}")
        self.kwargs = kwargs
        super().__init__()
        
        # parse_metadata with paths
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
        
        # need to be single channel audio, mean over all channels
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # if audio data is not equal to 30 secs, trim it or append it 
        if len(audio_data) != 30 * sr:
            if len(audio_data) < 30 * sr:
                audio_data = np.pad(audio_data, (0, 30 * sr - len(audio_data)))

            else:
                audio_data = audio_data[:30 * sr]
        feature = {'idx': idx, 'path': audio_fpath, 'array': audio_data}

        return feature 
    
    def __len__(self):
        return len(self.audio_fpaths)

def main(args):
    print(args) 

    accelerator = None
    accelerator = Accelerator(

    )
    dataset = PreFilterASRDataset(args.manifest)
    if accelerator.is_main_process:
        print(f"Total samples: {len(dataset)}")
        
    # TODO: here need to be whipser-base, batch_size=64
    hallucination_dectector = WhisperSmallModelDetector(small_model_card='openai/whisper-tiny', accelerator=accelerator)
    dataloader_for_test = DataLoader(dataset, 
        batch_size=16, 
        num_workers=8,
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
    
    # get rank by os, if distributed system used
    # rank = int(os.environ['RANK'])
    # print(f"Rank: {rank}")

    accelerator.wait_for_everyone()
    with open(f"{args.output_dir}/validator_inference.txt", "w") as fw:
        for i, batch in enumerate(dataloader_for_test):

            idxs, preds_str = hallucination_dectector.generate(batch)
            assert len(idxs) == len(preds_str), f"Length of idxs and pred_str should be the same, current idxs: {len(idxs)}, pred_str: {len(preds_str)}"
            steps_inference_progress_bar.update(1)
            
            # gather all results
            for idx, pred_str in zip(idxs, preds_str):
                fw.write(f"{idx}\t{pred_str}\n")
                fw.flush()
                
    print("Everthing is done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--validator_model_card",
        default="openai/whisper-base",
        help="a sample arg",
    )
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
    parser.add_argument(
        "--output_dir",
        default=None,
        required=True,
        help="a sample arg",
    )
    args = parser.parse_args()

    main(args)