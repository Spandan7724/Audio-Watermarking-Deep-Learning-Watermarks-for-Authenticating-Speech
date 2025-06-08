import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from scipy.stats import kurtosis
from scipy.signal import butter, lfilter
import glob
import warnings
from functools import partial

############################################## USAGE EXAMPLE ################################################
# python noise_mul.py --dir /path/to/audio/files --output /path/to/output --workers 4 --max_files 1000 --chunk_size 50 --sample_rate 16000
# python noise_mul.py --dir data/200_speech_only --output ./teststst_200_classification_results --sample 10000  --workers 16
#############################################################################################################

# Suppress warnings to speed up processing
warnings.filterwarnings("ignore")

def analyze_audio_file(file_path, sr=16000, compute_all_features=False):
    """
    Analyze a single audio file and return features that help determine if it's speech or noise.
    Optimized for speed by using a lower sampling rate and computing only essential features.
    """
    try:
        # Load audio file with reduced sampling rate
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        
        # Calculate basic features
        results = {
            'file_path': file_path,
            'duration': len(y) / sr,  # Faster than librosa.get_duration
        }
        
        # Feature 1: Energy statistics (simplified)
        energy = np.mean(y**2)
        results['energy'] = energy
        
        # Apply bandpass filter focusing on speech frequencies (simplified)
        def butter_bandpass(lowcut, highcut, fs, order=3):  # Reduced order for speed
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return b, a
        
        def bandpass_filter(data, lowcut, highcut, fs, order=3):  # Reduced order for speed
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            y = lfilter(b, a, data)
            return y
        
        # Apply bandpass filter focusing on speech frequencies
        y_speech = bandpass_filter(y, 300, 3000, sr, order=3)
        speech_energy = np.mean(y_speech**2)
        results['speech_band_energy'] = speech_energy
        
        # Feature 2: Zero-crossing rate (optimized)
        zcr = np.mean(np.abs(np.diff(np.signbit(y).astype(int))))
        results['zero_crossing_rate'] = float(zcr)
        
        # Feature 3: Spectral centroid (only if needed)
        if compute_all_features:
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            results['spectral_centroid'] = float(spectral_centroid)
        else:
            # Approximation using FFT
            fft = np.abs(np.fft.rfft(y))
            freqs = np.fft.rfftfreq(len(y), 1/sr)
            results['spectral_centroid'] = np.sum(freqs * fft) / (np.sum(fft) + 1e-8)
        
        # Feature 4: Kurtosis (speech typically has higher kurtosis)
        results['kurtosis'] = float(kurtosis(y))
        
        # Feature 5: Energy variation over time (speech has more variation)
        # More efficient frame energy calculation
        hop_length = sr // 100  # 10ms hop
        frame_length = sr // 40  # 25ms frames
        
        # Use strided array for more efficient framing
        shape = ((len(y) - frame_length) // hop_length + 1, frame_length)
        strides = (y.strides[0] * hop_length, y.strides[0])
        frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
        
        frame_energies = np.mean(frames**2, axis=1)
        energy_std = np.std(frame_energies)
        results['energy_std'] = float(energy_std)
        
        # Feature 6: Speech-to-noise ratio estimation
        results['speech_to_noise_ratio'] = speech_energy / (energy + 1e-10)
        
        return results
    
    except Exception as e:
        return {'file_path': file_path, 'error': str(e)}

def classify_speech_noise(features):
    """
    Classify a file as speech or noise based on extracted features.
    Simplified for speed but maintains accuracy.
    """
    # Skip files with errors
    if 'error' in features:
        return 'error'
    
    # Initialize score (higher means more likely to be speech)
    speech_score = 0
    
    # 1. Energy in speech frequency band (300-3000 Hz) should be significant for speech
    if features['speech_band_energy'] > 0.001:
        speech_score += 1
    
    # 2. Speech typically has a lower zero-crossing rate than noise
    if features['zero_crossing_rate'] < 0.1:
        speech_score += 1
    
    # 3. Spectral centroid is usually lower for speech
    if features['spectral_centroid'] < 3000:
        speech_score += 1
    
    # 4. Kurtosis is typically higher for speech (more "peaky")
    if features['kurtosis'] > 5:
        speech_score += 1
    
    # 5. Energy variation over time is higher for speech
    if features['energy_std'] > 0.01:
        speech_score += 1
        
    # 6. Speech-to-noise ratio should be higher for speech
    if features['speech_to_noise_ratio'] > 0.6:
        speech_score += 2  # Give this double weight
    
    # Classify based on score
    if speech_score >= 4:
        return 'speech'
    else:
        return 'noise'

def process_files_chunk(files_chunk, sr=16000, compute_all_features=False):
    """Process a chunk of files and return results"""
    results = []
    for file_path in files_chunk:
        features = analyze_audio_file(file_path, sr=sr, compute_all_features=compute_all_features)
        features['classification'] = classify_speech_noise(features)
        results.append(features)
    return results

def process_audio_directory(directory_path, num_workers=16, max_files=None, chunk_size=100, sr=16000):
    """
    Process all WAV files in a directory and classify them as speech or noise.
    Optimized with chunking and faster file discovery.
    
    Args:
        directory_path: Path to directory containing WAV files
        num_workers: Number of parallel workers
        max_files: Maximum number of files to process (for testing)
        chunk_size: Number of files to process in each parallel chunk
        sr: Sample rate to use when loading audio (lower = faster)
    
    Returns:
        DataFrame with results
    """
    # Get all WAV files (faster using glob)
    print("Scanning directory for WAV files...")
    wav_files = glob.glob(os.path.join(directory_path, "**/*.wav"), recursive=True)
    
    # Limit files if needed
    if max_files and len(wav_files) > max_files:
        wav_files = wav_files[:max_files]
    
    total_files = len(wav_files)
    print(f"Found {total_files} WAV files")
    
    # Create chunks for parallel processing
    chunks = [wav_files[i:i + chunk_size] for i in range(0, len(wav_files), chunk_size)]
    print(f"Processing in {len(chunks)} chunks with {num_workers} workers")
    
    # Create a partial function with fixed parameters
    process_func = partial(process_files_chunk, sr=sr, compute_all_features=False)
    
    # Process chunks in parallel
    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for chunk_results in tqdm(executor.map(process_func, chunks), total=len(chunks)):
            all_results.extend(chunk_results)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Calculate statistics
    stats = {
        'total_files': len(df),
        'speech_files': sum(df['classification'] == 'speech'),
        'noise_files': sum(df['classification'] == 'noise'),
        'error_files': sum(df['classification'] == 'error')
    }
    
    print("\nClassification Results:")
    print(f"Total files analyzed: {stats['total_files']}")
    print(f"Speech files: {stats['speech_files']} ({stats['speech_files']/stats['total_files']*100:.1f}%)")
    print(f"Noise files: {stats['noise_files']} ({stats['noise_files']/stats['total_files']*100:.1f}%)")
    
    if stats['error_files'] > 0:
        print(f"Files with errors: {stats['error_files']} ({stats['error_files']/stats['total_files']*100:.1f}%)")
    
    return df, stats

def export_results(df, output_dir):
    """
    Export classification results and create lists of speech/noise files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Export full results
    df.to_csv(os.path.join(output_dir, 'audio_classification_results.csv'), index=False)
    
    # Export lists of files
    speech_files = df[df['classification'] == 'speech']['file_path'].tolist()
    noise_files = df[df['classification'] == 'noise']['file_path'].tolist()
    
    with open(os.path.join(output_dir, 'speech_files.txt'), 'w') as f:
        f.write('\n'.join(speech_files))
    
    with open(os.path.join(output_dir, 'noise_files.txt'), 'w') as f:
        f.write('\n'.join(noise_files))
    
    print(f"\nResults exported to {output_dir}")
    
    # Print final summary with clear formatting
    total_files = len(df)
    speech_count = len(speech_files)
    noise_count = len(noise_files)
    error_count = sum(df['classification'] == 'error')
    
    print("\n" + "="*50)
    print("FINAL CLASSIFICATION SUMMARY")
    print("="*50)
    print(f"TOTAL FILES ANALYZED:      {total_files:,}")
    print(f"SPEECH FILES:              {speech_count:,} ({speech_count/total_files*100:.2f}%)")
    print(f"NOISE FILES:               {noise_count:,} ({noise_count/total_files*100:.2f}%)")
    if error_count > 0:
        print(f"FILES WITH ERRORS:         {error_count:,} ({error_count/total_files*100:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='Analyze audio files to classify as speech or noise')
    parser.add_argument('--dir', required=True, help='Directory containing WAV files')
    parser.add_argument('--output', default='./classification_results', help='Output directory for results')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='Number of parallel workers')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum number of files to process (for testing)')
    parser.add_argument('--chunk_size', type=int, default=50, help='Number of files to process in each parallel chunk')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate to use when loading audio')
    parser.add_argument('--sample', type=int, default=0, help='Number of random files to analyze for tuning (use 0 to skip)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.sample > 0:
        # Run on a sample to help tune parameters
        print(f"Analyzing {args.sample} random files for parameter tuning...")
        wav_files = glob.glob(os.path.join(args.dir, "**/*.wav"), recursive=True)
        
        if len(wav_files) > args.sample:
            import random
            wav_files = random.sample(wav_files, args.sample)
        
        sample_results = []
        for file in tqdm(wav_files):
            features = analyze_audio_file(file, sr=args.sample_rate, compute_all_features=True)
            classification = classify_speech_noise(features)
            features['classification'] = classification
            sample_results.append(features)
        
        sample_df = pd.DataFrame(sample_results)
        print("\nFeature statistics for tuning:")
        print(sample_df.describe())
        
        # Save sample results for manual inspection
        sample_output = os.path.join(args.output, 'sample_analysis')
        os.makedirs(sample_output, exist_ok=True)
        sample_df.to_csv(os.path.join(sample_output, 'sample_features.csv'), index=False)
        print(f"Sample analysis saved to {sample_output}")
    
    # Run full analysis
    print("\nStarting full analysis...")
    with tqdm(total=100, desc="Overall Progress", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        pbar.update(5)  # Update for initialization
        df, stats = process_audio_directory(
            args.dir, 
            num_workers=args.workers, 
            max_files=args.max_files,
            chunk_size=args.chunk_size,
            sr=args.sample_rate
        )
        pbar.update(85)  # Update for processing completion
        export_results(df, args.output)
        pbar.update(10)  # Update for export completions
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
    print(f"Average processing time per file: {elapsed_time/stats['total_files']:.4f} seconds")