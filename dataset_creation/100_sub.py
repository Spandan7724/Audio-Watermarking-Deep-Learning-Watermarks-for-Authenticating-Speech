import os
import librosa
import numpy as np
import soundfile as sf
import csv
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial

def preprocess_audio_file(filepath, output_dir, target_sr=16000, segment_duration=1.0):
    """
    Processes a single audio file: loads, resamples, normalizes, and splits into segments.
    An inner tqdm progress bar shows segmentation progress.
    """
    try:
        # Load audio at its original sampling rate.
        audio, sr = librosa.load(filepath, sr=None)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # Normalize audio (scale max amplitude to 0.99).
        audio = audio / np.max(np.abs(audio)) * 0.99

        # Calculate segment details.
        segment_samples = int(segment_duration * sr)
        total_samples = len(audio)
        num_segments = total_samples // segment_samples

        # Create output directory for this file.
        base_filename = os.path.splitext(os.path.basename(filepath))[0]
        file_output_dir = os.path.join(output_dir, base_filename)
        os.makedirs(file_output_dir, exist_ok=True)

        # Process each segment with an inner progress bar.
        for i in tqdm(range(num_segments), 
                      desc=f"Segmenting {base_filename}", 
                      unit="segment", 
                      leave=False):
            start = i * segment_samples
            end = start + segment_samples
            segment = audio[start:end]
            segment_filename = os.path.join(file_output_dir, f"{base_filename}_seg{i+1}.wav")
            sf.write(segment_filename, segment, sr)

        return num_segments
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return 0

def process_row(row, output_dir, target_sr, segment_duration):
    """
    Processes a single row from the CSV.
    """
    filepath = row["filepath"]
    segments = preprocess_audio_file(filepath, output_dir, target_sr, segment_duration)
    return segments

def process_files_parallel(subset_csv, output_dir, target_sr=16000, segment_duration=1.0, max_workers=4):
    """
    Reads the CSV file, and processes each audio file in parallel using process_map.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV to get a list of rows.
    with open(subset_csv, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    # Create a partial function that fixes extra parameters.
    process_func = partial(process_row, output_dir=output_dir, 
                           target_sr=target_sr, segment_duration=segment_duration)

    # Use process_map to apply process_func over rows in parallel.
    # process_map displays its own progress bar.
    results = process_map(process_func, rows, max_workers=max_workers)

    total_segments = sum(results)
    print(f"Preprocessing complete. Processed {len(rows)} files and generated {total_segments} segments.")

if __name__ == "__main__":
    subset_csv = "subset_100hours.csv"
    output_directory = "data/100/preprocessed_audio"
    # Adjust max_workers as needed.
    process_files_parallel(subset_csv, output_directory, max_workers=4)