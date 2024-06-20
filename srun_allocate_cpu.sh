image_root=/mnt/home/andybi7676/images
image_tag=v0.0.3
cpu_num=$1

image_fpath=$image_root/ntu-cool-asr_$image_tag.sqsh

echo "try to allocate $cpu_num cpus, with image=$image_fpath"

srun -N 1 --cpus-per-task=$cpu_num \
    --container-image $image_fpath \
    --container-mounts /mnt/home/andybi7676/corpus:/work/corpus,/mnt/home/andybi7676/distil-whisper:/workspace/distil-whisper \
    --container-writable --no-container-mount-home --pty bash