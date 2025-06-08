import os
import shutil

# Path to the file containing list of speech file paths
speech_files_path = '200_classification_results/speech_files.txt'

# Destination folder for speech files
destination_folder = os.path.join('data', '200_speech_only')
os.makedirs(destination_folder, exist_ok=True)

# Open the speech_files.txt and iterate over each file path
with open(speech_files_path, 'r') as f:
    for line in f:
        file_path = line.strip()  # Remove any leading/trailing whitespace/newlines
        if os.path.isfile(file_path):
            try:
                shutil.copy(file_path, destination_folder)
                print(f"Copied: {file_path}")
            except Exception as e:
                print(f"Error copying {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
