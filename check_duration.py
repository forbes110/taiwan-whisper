from pydub import AudioSegment

# 读取音频文件 (支持 .m4a 格式)
audio = AudioSegment.from_file("/mnt/dataset_1T/TaiwanBar/2qqQIJlJc70.m4a")

# 获取音频的时长（毫秒）
duration_milliseconds = len(audio)

# 将时长转换为秒
duration_seconds = duration_milliseconds / 1000

# 获取音频的采样率和通道数
sample_rate = audio.frame_rate
channels = audio.channels

# 打印音频信息
print(f"Duration: {duration_seconds} seconds")
print(f"Sample rate: {sample_rate} Hz")
print(f"Channels: {channels}")

# 如果需要转换为 NumPy 数组：
import numpy as np

# 提取音频的原始数据为字节，并转换为 NumPy 数组
audio_samples = np.array(audio.get_array_of_samples())

# 如果是立体声（2通道），需要重塑为二维数组
if channels == 2:
    audio_samples = audio_samples.reshape((-1, 2))

print(f"Audio data shape: {audio_samples.shape}")
