from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import argparse
import os.path as osp
import os
import re


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Remove common hallucination from transcriptions.")
    parser.add_argument("--original_tsv", type=str, default="/mnt/audio_paths.tsv", help="Path to tsv file of audios.")
    parser.add_argument("--output_dir", type=str, default="/mnt/common_hallucination_caught", help="Path to tsv file of audios.")
    return parser.parse_args()
def main():
    args = parse_args()
    
    output_dir = args.output_dir
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    hallucination_match_list = [
        "Okay.",
        "...",
        ".",
        "Mm.",
    ]

    hallucination_contain_list = [
        "請不吝",
        r"(?<!\w)org(?!\w)", # ensure the examples like "organization" are not removed
        "點贊",
        "點讚",
        "字幕提供",
        "支持明鏡",
        "點點欄目"
    ]
    # Initialize the lists to store results
    contain_results = []
    match_results = []

    normalizer = BasicTextNormalizer()

    # load original tsv for audio paths
    with open(args.original_tsv, "r") as f:
        root = f.readline().strip()
        audio_subfpaths = [l.strip() for l in f.readlines()]
        audio_fpaths = [osp.join(root, audio_subfpath) for audio_subfpath in audio_subfpaths]
        trans_fpaths = [audio_fpath.replace('flac', 'txt') for audio_fpath in audio_fpaths]
    
    for trans_fpath in trans_fpaths:
        with open(trans_fpath, "r") as f:
        
            lines = f.readlines()
            
            # remove <|endoftext|>
            whisper_transcript = lines[0].strip().split("<|endoftext|>")[0]
            
            # remove <|continued|>
            whisper_transcript = whisper_transcript.split("<|continued|>")[0] 
            
            # prev segment as prompt
            end_transcript = lines[1].strip()
            
            # find all timestamp tokens ('<|*|>') and remove them in the transcript
            timestamp_tokens = re.findall(r"<\|\d{1,2}\.\d{2}\|>", whisper_transcript + end_transcript)
            
            for st in timestamp_tokens:
                whisper_transcript = whisper_transcript.replace(st, ' ')
            whisper_transcript = whisper_transcript.strip().replace('  ', ' ')
            
            # standardize the transcript
            whisper_transcript = normalizer(whisper_transcript)
            
            # TODO: Note that we just collect, not remove, need to check further
            
            # Check for contains cases
            matched_keywords = [keyword for keyword in hallucination_contain_list if keyword in whisper_transcript]
            if matched_keywords:
                contain_results.append({
                    "trans_fpath": trans_fpath,
                    "matched_keywords": ', '.join(matched_keywords),
                    "transcript": whisper_transcript
                })

            # Check for match cases
            words = re.findall(r'\b\w+\b|\.\.\.|[^\s\w]', whisper_transcript)
            matched_words = [word for word in words if word in hallucination_match_list]
            if matched_words:
                match_results.append({
                    "trans_fpath": trans_fpath,
                    "matched_words": ', '.join(matched_words),
                    "transcript": whisper_transcript
                })

    # Save results to CSV with UTF-8 encoding
    if contain_results:
        pd.DataFrame(contain_results).to_csv(f"{output_dir}/contain_cases.csv", index=False, encoding="utf-8")
    if match_results:
        pd.DataFrame(match_results).to_csv(f"{output_dir}/match_cases.csv", index=False, encoding="utf-8")

    if not match_results and not contain_results:
        print("No common hallucination found.")
                                
if __name__ == "__main__":
    main()
    