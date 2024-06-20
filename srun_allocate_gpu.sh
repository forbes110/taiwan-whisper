image_root=/mnt/home/andybi7676/images
image_tag=v0.0.3
gpu_num=$1

image_fpath=$image_root/ntu-cool-asr_$image_tag.sqsh

echo "try to allocate $gpu_num gpus, with image=$image_fpath"

srun -N 1 --cpus-per-task=64 -G $gpu_num \
    --container-image $image_root/ntu-cool-asr_$image_tag.sqsh \
    --container-mounts /mnt/home/andybi7676/corpus:/work/corpus,/mnt/home/andybi7676/distil-whisper:/workspace/distil-whisper \
    --container-writable --no-container-mount-home --pty bash