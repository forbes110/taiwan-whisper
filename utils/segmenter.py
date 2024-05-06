import torch
import torchaudio
from torchaudio.transforms import VAD

# Load the audio file
audio_file = "path/to/your/audio/file.wav"
waveform, sample_rate = torchaudio.load(audio_file)

# Create the VAD transform
vad = VAD(sample_rate=sample_rate)

# Perform voice activity detection
segments = vad(waveform)

# Print the start and end times of each segment
for segment in segments:
    start_time = segment[0] / sample_rate
    end_time = segment[1] / sample_rate
    print(f"Segment: {start_time:.2f}s - {end_time:.2f}s")

# Save each segment as a separate audio file
for i, segment in enumerate(segments):
    start_sample = segment[0]
    end_sample = segment[1]
    segment_waveform = waveform[:, start_sample:end_sample]
    torchaudio.save(f"segment_{i}.wav", segment_waveform, sample_rate)