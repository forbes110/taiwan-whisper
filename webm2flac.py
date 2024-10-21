import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def convert_webm_to_flac(input_file, output_file):
    """Converts a WebM file to FLAC format."""
    command = [
        'ffmpeg',
        '-i', input_file,
        '-vn',  # Disable video stream
        '-acodec', 'flac',  # Use FLAC codec
        output_file
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Conversion successful: {input_file} -> {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_file}: {e}")

def process_file_pair(file_pair):
    """Helper function to process a single file pair in multi-threading."""
    input_file, output_file = file_pair
    convert_webm_to_flac(input_file, output_file)

def batch_convert_webm_to_flac(input_directory, output_directory, max_workers=4):
    """Batch converts all WebM files in a directory to FLAC using multi-threading."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  # Create output directory if it doesn't exist

    # Prepare file pairs (input and output paths)
    file_pairs = [
        (
            os.path.join(input_directory, filename),
            os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.flac")
        )
        for filename in os.listdir(input_directory) if filename.endswith('.webm')
    ]

    # Use ThreadPoolExecutor to convert files concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file_pair, pair) for pair in file_pairs]

        for future in as_completed(futures):
            try:
                future.result()  # Check for exceptions
            except Exception as e:
                print(f"Error occurred: {e}")

# Usage
input_dir = '/mnt/dataset_1T/FTV_News/'
output_dir = '/mnt/dataset_1T/FTV_News_flac/'
batch_convert_webm_to_flac(input_dir, output_dir, max_workers=8)  # Adjust max_workers based on CPU cores
