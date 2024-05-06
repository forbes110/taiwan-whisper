#!/usr/bin/bash

# Add your code here
ROOT=/work/data/test
subdirs=$(ls $ROOT)
echo "Testing subdirs: $subdirs"

# reformating the raw data and create a manifest.
# for subdir in $subdirs; do
#     echo "Processing $subdir"
#     python prepare_test_data.py $ROOT/$subdir # special .py file for preparing COOL testing data. 
# done

# segment the audio data by .srt files
for subdir in $subdirs; do
    echo "Segmenting $subdir based on .srt files"
    python segment_audio.py --audio_dir $ROOT/$subdir/raw/audio \
        --srt_dir $ROOT/$subdir/raw/revised \
        --output_dir $ROOT/$subdir/segmented
done

