import os
import glob
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

TARGET_SAMPLE_RATE = 16000  # Target sample rate

def resample_and_convert_to_flac(input_path):
    """
    Check the sample rate of the audio file using pydub.
    If it is not 16000 Hz, resample it and convert to FLAC.

    Args:
        input_path (str): Path to the input audio file

    Returns:
        str: Success message if the conversion succeeded, otherwise an error message
    """
    try:
        # Load audio file using pydub
        audio = AudioSegment.from_file(input_path)
        
        # Check the sample rate of the audio file

        # Resample the audio if needed
        if audio.frame_rate != TARGET_SAMPLE_RATE:
            # Resample the audio if needed
            print(f"Resampling {input_path} from {audio.frame_rate} to {TARGET_SAMPLE_RATE} Hz")
            audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
        

        # Generate new filename with .flac extension
        flac_path = input_path.replace(".m4a", ".flac")
        

        # Export the audio to FLAC format
        audio.export(flac_path, format="flac")

        return f"Converted and saved: {flac_path}"
    
    except Exception as e:
        return f"Error processing {input_path}: {e}"

def process_directory(directory, max_workers=4):
    """
    Recursively find all .m4a files in the directory, resample, and convert them to FLAC,
    using multi-threading and a progress bar.
    """
    # Find all .m4a files recursively
    m4a_files = glob.glob(os.path.join(directory, '**', '*.m4a'), recursive=True)

    if not m4a_files:
        print("No .m4a files found.")
        return

    print(f"Found {len(m4a_files)} .m4a files. Processing...")

    # Use ThreadPoolExecutor for multi-threading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        futures = {executor.submit(resample_and_convert_to_flac, file): file for file in m4a_files}

        # Use tqdm to display progress bar as tasks complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing audio files"):
            result = future.result()
            print(result)

if __name__ == "__main__":
    # Example usage: Change 'directory' to your target path
    directory = "/mnt/dataset_1T/"
    process_directory(directory, max_workers=8)  # Adjust number of threads if needed