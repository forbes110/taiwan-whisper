# import os
# import glob
# import argparse
# from pydub import AudioSegment
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm

# TARGET_SAMPLE_RATE = 16000  # Target sample rate

# def resample_and_convert_to_flac(input_path):
#     """
#     Check the sample rate of the audio file using pydub.
#     If it is not 16000 Hz, resample it and convert to FLAC.

#     Args:
#         input_path (str): Path to the input audio file

#     Returns:
#         str: Success message if the conversion succeeded, otherwise an error message
#     """
#     try:
#         # Load audio file using pydub
#         audio = AudioSegment.from_file(input_path)
        
#         # Check the sample rate of the audio file
#         if audio.frame_rate != TARGET_SAMPLE_RATE:
#             print(f"Resampling {input_path} from {audio.frame_rate} to {TARGET_SAMPLE_RATE} Hz")
#             audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)

#         # Generate new filename with .flac extension
#         flac_path = input_path.replace(".m4a", ".flac")

#         # Export the audio to FLAC format
#         audio.export(flac_path, format="flac")

#         return f"Converted and saved: {flac_path}"
    
#     except Exception as e:
#         return f"Error processing {input_path}: {e}"

# def process_directory(directory, max_workers=4):
#     """
#     Recursively find all .m4a files in the directory, resample, and convert them to FLAC,
#     using multi-threading and a progress bar.
#     """
#     # Find all .m4a files recursively
#     m4a_files = glob.glob(os.path.join(directory, '**', '*.m4a'), recursive=True)

#     if not m4a_files:
#         print("No .m4a files found.")
#         return

#     print(f"Found {len(m4a_files)} .m4a files. Processing...")

#     # Use ThreadPoolExecutor for multi-threading
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = {executor.submit(resample_and_convert_to_flac, file): file for file in m4a_files}

#         # Use tqdm to display progress bar as tasks complete
#         for future in tqdm(as_completed(futures), total=len(futures), desc="Processing audio files"):
#             result = future.result()
#             print(result)

# def parse_args():
#     """
#     Parse command-line arguments.
#     """
#     parser = argparse.ArgumentParser(description="Resample and convert .m4a files to FLAC.")
#     parser.add_argument("--directory", type=str, default="/mnt/dataset_1T", help="Directory containing .m4a files.")
#     parser.add_argument("--max_workers", type=int, default=4, help="Number of threads for processing.")
#     return parser.parse_args()

# if __name__ == "__main__":
#     # Parse command-line arguments
#     args = parse_args()

#     # Process the specified directory
#     process_directory(args.directory, max_workers=args.max_workers)


import os
import glob
import argparse
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

TARGET_SAMPLE_RATE = 16000  # Target sample rate
CHUNK_SIZE = 100  # Process files in chunks for better memory management

def resample_and_convert_to_flac(input_path):
    """
    Resample audio file to 16kHz and convert to FLAC format.
    Delete the original m4a file if conversion is successful.
    """
    try:
        start_time = time.time()
        flac_path = input_path.replace(".m4a", ".flac")
        
        # Skip if FLAC file already exists
        if os.path.exists(flac_path):
            return f"Skipped (already exists): {flac_path}"
        
        # Load audio file using pydub with specific format
        audio = AudioSegment.from_file(input_path, format="m4a")
        
        # Only resample if necessary
        if audio.frame_rate != TARGET_SAMPLE_RATE:
            audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)

        # Export with optimized settings
        audio.export(
            flac_path,
            format="flac",
            parameters=["-compression_level", "8"]  # Optimize for speed
        )
        
        # Verify the new file exists and has size > 0
        if os.path.exists(flac_path) and os.path.getsize(flac_path) > 0:
            os.remove(input_path)  # Delete original m4a file
            processing_time = time.time() - start_time
            return f"Converted and cleaned up: {flac_path} ({processing_time:.2f}s)"
        else:
            return f"Error: FLAC file not created properly for {input_path}"
        
    except Exception as e:
        return f"Error processing {input_path}: {str(e)}"

def process_chunk(file_chunk, max_workers):
    """Process a chunk of files using ThreadPoolExecutor"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(resample_and_convert_to_flac, file): file 
                  for file in file_chunk}
        
        for future in as_completed(futures):
            result = future.result()
            logging.info(result)

def process_directory(directory, max_workers=None):
    """
    Process audio files in chunks using multiple processes and threads.
    """
    # Auto-configure number of workers if not specified
    if max_workers is None:
        max_workers = min(32, multiprocessing.cpu_count() * 2)
    
    # Find all .m4a files recursively
    m4a_files = glob.glob(os.path.join(directory, '**', '*.m4a'), recursive=True)
    
    if not m4a_files:
        logging.info("No .m4a files found.")
        return

    total_files = len(m4a_files)
    logging.info(f"Found {total_files} .m4a files. Processing with {max_workers} workers...")

    # Process files in chunks
    chunks = [m4a_files[i:i + CHUNK_SIZE] 
             for i in range(0, len(m4a_files), CHUNK_SIZE)]
    
    # Use ProcessPoolExecutor for chunk-level parallelism
    with multiprocessing.Pool(processes=min(len(chunks), multiprocessing.cpu_count())) as pool:
        list(tqdm(
            pool.starmap(
                process_chunk,
                [(chunk, max_workers) for chunk in chunks]
            ),
            total=len(chunks),
            desc="Processing chunks"
        ))

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Resample and convert .m4a files to FLAC with cleanup."
    )
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Directory containing .m4a files."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of worker threads (default: 2 * CPU cores)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    
    try:
        process_directory(args.directory, args.max_workers)
        total_time = time.time() - start_time
        logging.info(f"Processing completed in {total_time:.2f} seconds")
    except KeyboardInterrupt:
        logging.warning("\nProcessing interrupted by user")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")