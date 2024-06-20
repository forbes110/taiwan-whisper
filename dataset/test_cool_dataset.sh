root=~/distil-whisper
data_root=~/distil-whisper/corpus/data

cd $root
source path.sh
cd $root/dataset

# accelerate launch --main_process_port 29500 cool_dataset.py \
#     --manifest "$data_root/manifest/train-all_new.tsv"

# cp the first 10 audio files to the test folder
# mkdir -p ./test
# root_dir=$(head -n 1 $data_root/manifest/train-all_new.tsv | awk '{print $1}')
# echo "$root_dir is the root dir"
# for sub_fpath in $(tail -n +2 $data_root/manifest/train-all_new.tsv | head -n 10 | awk '{print $1}'); do
#     audio_fpath=$root_dir/$sub_fpath
#     # echo "cp $audio_fpath to ./test"
#     cp $audio_fpath ./test
#     text_fpath=$(echo $audio_fpath | sed 's/flac/txt/')
#     echo "cp $text_fpath to ./test"
#     cp $text_fpath ./test
# done

# concatenate all the text files and sort by index (use tab as separator)
cat inference_results/whisper_base/idx_hyp.*.txt | grep '\S' | sort -n > inference_results/whisper_base/idx_hyp.txt
