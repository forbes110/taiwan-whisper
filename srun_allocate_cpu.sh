srun -N 1 --cpus-per-task=64 \
    --container-image /mnt/home/andybi7676/ntu-cool-asr_v0.0.1.sqsh \
    --container-mounts /mnt/home/andybi7676/corpus:/work/corpus,/mnt/home/andybi7676/distil-whisper:/workspace/distil-whisper \
    --container-writable --no-container-mount-home --pty bash