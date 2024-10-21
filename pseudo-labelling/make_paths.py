import os
import csv
import argparse

def generate_paths_csvs(root_dir):
    all_audio_paths = []
    csv_files_created = 0

    # Generate individual CSV files for each directory
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        
        if os.path.isdir(dir_path):
            csv_path = os.path.join(root_dir, f"{dir_name}.csv")
            dir_audio_paths = []
            
            for filename in os.listdir(dir_path):
                if filename.endswith('.flac'):
                    audio_path = os.path.abspath(os.path.join(dir_path, filename))
                    dir_audio_paths.append([audio_path])
                    all_audio_paths.append([audio_path])
            
            # Write the audio paths to the current directory's CSV file
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["audio_path"])
                writer.writerows(dir_audio_paths)
            
            print(f"Generated {csv_path}")
            csv_files_created += 1

    # Generate a merged raw_data.tsv file
    merged_tsv_path = os.path.join(root_dir, "raw_data.tsv")
    with open(merged_tsv_path, 'w', newline='', encoding='utf-8') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerow(["audio_path"])
        writer.writerows(all_audio_paths)

    print(f"Generated {merged_tsv_path}")
    return csv_files_created

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CSV files of audio paths.")
    parser.add_argument(
        "--root_dir", type=str, default="/mnt/dataset", 
        help="Path to the root directory containing audio datasets."
    )
    args = parser.parse_args()

    total_csv_files = generate_paths_csvs(args.root_dir)
    print(f"Total CSV files generated: {total_csv_files}")
    
    
# for single dir, we use:
# echo -e "audio_paths" > audio_files.tsv
# find "$(pwd)" -name "*.flac" >> audio_files.tsv


# echo -e "audio_paths" > audio_files.tsv
# find "$(pwd)" -name "*.m4a" >> audio_files.tsv
