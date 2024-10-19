import soundfile as sf

# Replace 'your_audio_file.flac' with the path to your FLAC file
data, samplerate = sf.read('/mnt/data_pair/BN1YP9VIB08/BN1YP9VIB08_4320-435680.flac')

# Calculate duration in seconds
duration_seconds = len(data) / samplerate

print(f"Duration: {duration_seconds} seconds")
