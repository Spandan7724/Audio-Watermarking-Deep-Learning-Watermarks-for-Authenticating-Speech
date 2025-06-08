import os
import glob
import torchaudio
import torch
from tqdm import tqdm

def is_silent(waveform, threshold=1e-4):
    """
    Check if a waveform is silent based on RMS energy.
    
    Args:
        waveform (torch.Tensor): Audio tensor of shape (channels, samples).
        threshold (float): RMS energy threshold below which we consider the file silent.
        
    Returns:
        bool: True if silent, False otherwise.
    """
    # Compute RMS energy
    rms = torch.sqrt(torch.mean(waveform ** 2))
    return rms < threshold

def count_silent_files(root_dir, threshold=1e-4):
    """
    Scan a directory for .wav files and count how many are silent.

    Args:
        root_dir (str): Path to dataset folder containing .wav files.
        threshold (float): RMS energy threshold for silence detection.

    Returns:
        int: Number of silent files.
    """
    filepaths = glob.glob(os.path.join(root_dir, '**', '*.wav'), recursive=True)
    silent_count = 0
    total_count = len(filepaths)

    if total_count == 0:
        print("No .wav files found in the directory.")
        return

    print(f"Checking {total_count} .wav files for silence...")

    # Use tqdm progress bar
    for filepath in tqdm(filepaths, desc="Processing Files", unit="file"):
        waveform, sr = torchaudio.load(filepath)
        
        # Convert to mono if stereo/multi-channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Check for silence
        if is_silent(waveform, threshold):
            silent_count += 1
            print(f"Silent: {filepath}")

    print(f"\nTotal files checked: {total_count}")
    print(f"Silent files: {silent_count} ({(silent_count/total_count)*100:.2f}%)")

# Usage
data_root = "data/100_all" 
count_silent_files(data_root, threshold=1e-4)
