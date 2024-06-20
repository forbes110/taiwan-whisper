# This script is used to create simple manifest for cool asr testing set
# columns: audio_fpath, text
import os
import os.path as osp
import argparse
import soundfile as sf
from glob import glob
from tqdm import tqdm
from collections import defaultdict


def test_hf_dataset(tsv_fpath, hf_output_dir=None, split='test'):
    from datasets import Dataset, Audio, load_dataset, DatasetDict
    import pandas as pd
    df = pd.read_csv(tsv_fpath, sep="\t")
    # read from pd's df and convert to iterable dataset
    dataset = Dataset.from_pandas(df)
    # dataset = dataset.to_iterable_dataset()
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    print(dataset)
    # test for n=5
    for i, data in enumerate(dataset):
        if i == 5:
            break
        print(data)
    # save to disk if output_dir is provided
    if hf_output_dir is not None:
        dataset_dict = DatasetDict()
        dataset_dict['test'] = dataset
        dataset_dict.save_to_disk(hf_output_dir, num_proc=16)
        print(f"HF Dataset saved at: {hf_output_dir}")
        # test by loading dataset
        dataset = load_dataset(hf_output_dir, 'default', split=split, streaming=True)
        print(f"Loaded dataset: {dataset}, type={type(dataset)}")
        for i, data in enumerate(dataset):
            if i == 5:
                break
            print(data)


def main(args):
    # get all audio files
    audio_files = glob(osp.join(args.audio_dir, "**", "*.flac"), recursive=True)
    # sort by id
    cource_to_audio_fpath = defaultdict(list)
    for audio_fpath in audio_files:
        cource_id = osp.dirname(audio_fpath)
        cource_to_audio_fpath[cource_id].append(audio_fpath)
    audio_files = []
    for cource_id, audio_fpaths in cource_to_audio_fpath.items():
        audio_files.extend(sorted(audio_fpaths, key=lambda x: int(osp.basename(x).split(".")[0])))
    print(f"Found {len(audio_files)} audio files")
    # create manifest
    output_fpath = osp.join(args.output_dir, "cool_asr_test.tsv")
    total_duration_in_secs = 0
    with open(output_fpath, "w", encoding="utf-8") as fw:
        # write column names
        fw.write("audio\ttext\n")
        for audio_fpath in tqdm(audio_files):
            # try to load audio file
            info = sf.info(audio_fpath)
            assert info.samplerate == 16000, f"Invalid samplerate: {info.samplerate}"
            total_duration_in_secs += info.duration
            text_fpath = audio_fpath.replace(".flac", ".txt")
            if not osp.exists(text_fpath):
                print(f"Text file {text_fpath} not found for {audio_fpath}")
                continue
            with open(text_fpath, 'r') as fr:
                trans = fr.readline().strip()
            fw.write(f"{audio_fpath}\t{trans}\n")
    # log total duration
    print(f"Total duration: {total_duration_in_secs} seconds")
    print(f"Manifest file saved at: {output_fpath}")
    print("Done!")
    # test by hf dataset
    test_hf_dataset(output_fpath, hf_output_dir=args.hf_output_dir, split='test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create manifest for cool asr testing set")
    parser.add_argument("audio_dir", type=str, help="Directory containing audio files")
    parser.add_argument("output_dir", type=str, help="Output manifest file path")
    parser.add_argument("--hf_output_dir", default=None, help="Output directory for hf dataset")
    args = parser.parse_args()
    main(args)