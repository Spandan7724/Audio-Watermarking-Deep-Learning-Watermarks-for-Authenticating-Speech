import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from scipy.stats import kurtosis
from scipy.signal import butter, lfilter

def analyze_audio_file(file_path):
    """
    Analyze a single audio file and return features that help determine if it's speech or noise.
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Calculate features
        results = {
            'file_path': file_path,
            'duration': librosa.get_duration(y=y, sr=sr),
        }
        
        # Feature 1: Energy statistics
        energy = np.sum(y**2) / len(y)
        results['energy'] = energy
        
        # Feature a filter to focus on speech frequencies (300-3000 Hz)
        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return b, a
        
        def bandpass_filter(data, lowcut, highcut, fs, order=5):
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            y = lfilter(b, a, data)
            return y
        
        # Apply bandpass filter focusing on speech frequencies
        y_speech = bandpass_filter(y, 300, 3000, sr, order=5)
        speech_energy = np.sum(y_speech**2) / len(y_speech)
        results['speech_band_energy'] = speech_energy
        
        # Feature 2: Zero-crossing rate (typically higher for noise)
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        results['zero_crossing_rate'] = float(zcr)
        
        # Feature 3: Spectral centroid (higher for noise, lower for speech)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        results['spectral_centroid'] = float(spectral_centroid)
        
        # Feature 4: Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        results['spectral_bandwidth'] = float(spectral_bandwidth)
        
        # Feature 5: Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        results['rolloff'] = float(rolloff)
        
        # Feature 6: MFCC statistics (good for distinguishing speech from noise)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_vars = np.var(mfccs, axis=1)
        results['mfcc_mean'] = float(np.mean(mfcc_means))
        results['mfcc_var'] = float(np.mean(mfcc_vars))
        
        # Feature 7: Kurtosis (speech typically has higher kurtosis)
        results['kurtosis'] = float(kurtosis(y))
        
        # Feature 8: Energy variation over time (speech has more variation)
        # Split the signal into frames and calculate energy per frame
        frame_length = int(sr * 0.025)  # 25 ms frames
        hop_length = int(sr * 0.010)    # 10 ms hop
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        frame_energies = np.sum(frames**2, axis=0) / frame_length
        energy_std = np.std(frame_energies)
        results['energy_std'] = float(energy_std)
        
        # Feature 9: Speech-to-noise ratio estimation (using speech band vs overall energy)
        results['speech_to_noise_ratio'] = speech_energy / (energy + 1e-10)
        
        return results
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return {'file_path': file_path, 'error': str(e)}

def classify_speech_noise(features):
    """
    Classify a file as speech or noise based on extracted features.
    
    This is a heuristic-based classifier. For a more accurate approach, you could:
    1. Label a subset of your data manually
    2. Train a classifier (e.g., Random Forest, SVM)
    3. Use that classifier on the full dataset
    
    The current approach uses some common audio characteristics to distinguish speech from noise.
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

def process_audio_directory(directory_path, num_workers=16, max_files=None):
    """
    Process all WAV files in a directory and classify them as speech or noise.
    
    Args:
        directory_path: Path to directory containing WAV files
        num_workers: Number of parallel workers
        max_files: Maximum number of files to process (for testing)
    
    Returns:
        DataFrame with results
    """
    # Get all WAV files
    print("Scanning directory for WAV files...")
    wav_files = []
    for root, _, files in tqdm(os.walk(directory_path)):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    # Limit files if needed
    if max_files and len(wav_files) > max_files:
        wav_files = wav_files[:max_files]
    
    total_files = len(wav_files)
    print(f"Found {total_files} WAV files")
    
    # Process files in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(analyze_audio_file, wav_files), total=total_files):
            results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Add classification
    if 'error' not in df.columns:
        df['classification'] = df.apply(classify_speech_noise, axis=1)
    else:
        # Handle files with errors
        df['classification'] = df.apply(
            lambda row: 'error' if pd.notna(row.get('error')) else classify_speech_noise(row), 
            axis=1
        )
    
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
    
    parser = argparse.ArgumentParser(description='Analyze audio files to classify as speech or noise')
    parser.add_argument('--dir', required=True, help='Directory containing WAV files')
    parser.add_argument('--output', default='./classification_results', help='Output directory for results')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum number of files to process (for testing)')
    parser.add_argument('--sample', type=int, default=100, help='Number of random files to analyze for tuning (use 0 to skip)')
    
    args = parser.parse_args()
    
    if args.sample > 0:
        # Run on a sample to help tune parameters
        print(f"Analyzing {args.sample} random files for parameter tuning...")
        wav_files = [os.path.join(root, file) 
                    for root, _, files in os.walk(args.dir) 
                    for file in files if file.lower().endswith('.wav')]
        
        if len(wav_files) > args.sample:
            import random
            wav_files = random.sample(wav_files, args.sample)
        
        sample_results = []
        for file in tqdm(wav_files):
            features = analyze_audio_file(file)
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
        df, stats = process_audio_directory(args.dir, num_workers=args.workers, max_files=args.max_files)
        pbar.update(85)  # Update for processing completion
        export_results(df, args.output)
        pbar.update(10)  # Update for export completions