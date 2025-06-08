import os
import librosa
import numpy as np
import soundfile as sf
import csv

def preprocess_audio_file(filepath, output_dir, target_sr=16000, segment_duration=1.0):
    try:
        # Load audio using librosa
        audio, sr = librosa.load(filepath, sr=None)  # load at original sampling rate
        # Resample to target_sr if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        # Normalize the audio (e.g., to have max amplitude 0.99)
        audio = audio / np.max(np.abs(audio)) * 0.99
        
        # Calculate number of samples per segment
        segment_samples = int(segment_duration * sr)
        total_samples = len(audio)
        num_segments = total_samples // segment_samples
        
        # Create output directory for the file if not exists
        base_filename = os.path.splitext(os.path.basename(filepath))[0]
        file_output_dir = os.path.join(output_dir, base_filename)
        os.makedirs(file_output_dir, exist_ok=True)
        
        for i in range(num_segments):
            start = i * segment_samples
            end = start + segment_samples
            segment = audio[start:end]
            segment_filename = os.path.join(file_output_dir, f"{base_filename}_seg{i+1}.wav")
            sf.write(segment_filename, segment, sr)
        
        return num_segments
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return 0

def preprocess_subset(subset_csv, output_dir, target_sr=16000, segment_duration=1.0):
    """
    Reads the subset CSV, processes each file, and outputs segmented WAV files.
    
    Args:
        subset_csv (str): CSV file containing filepaths for the 100-hour subset.
        output_dir (str): Directory where preprocessed files will be stored.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(subset_csv, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        total_segments = 0
        file_count = 0
        for row in reader:
            filepath = row["filepath"]
            segments = preprocess_audio_file(filepath, output_dir, target_sr, segment_duration)
            total_segments += segments
            file_count += 1
            if file_count % 100 == 0:
                print(f"Processed {file_count} files, {total_segments} segments generated so far.")
    
    print(f"Preprocessing complete. Processed {file_count} files and generated {total_segments} segments.")

if __name__ == "__main__":
    subset_csv = "subset_100hours.csv"
    output_directory = "data/100/preprocessed_audio"
    preprocess_subset(subset_csv, output_directory)
