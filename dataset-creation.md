# Dataset Creation Guide for Audio Watermarking

This comprehensive guide provides step-by-step instructions for downloading, setting up, and processing audio datasets for training the audio watermarking system. The pipeline transforms raw audio files into processed 1-second segments optimized for model training.



## Overview

The audio watermarking system requires high-quality speech datasets for training. The primary workflow involves:

1. **Download**: Acquire raw audio datasets (VoxPopuli, LibriSpeech)
2. **Metadata**: Generate file inventories with duration information
3. **Selection**: Create targeted subsets (100h, 200h, test sets)
4. **Preprocessing**: Convert to 1-second WAV segments at 16kHz
5. **Classification**: Filter speech vs. noise content
6. **Quality Control**: Remove silent or low-quality segments

##  Prerequisites



### Software Dependencies
```bash
# Core dependencies
pip install torch torchaudio matplotlib tqdm numpy pandas

# Dataset processing
pip install datasets soundfile librosa scipy

# Optional audio metrics
pip install pesq pystoi sklearn seaborn
```

### Python Environment Setup
```bash
# Create virtual environment
python -m venv audio_watermark_env
source audio_watermark_env/bin/activate  # Linux/Mac
# or
audio_watermark_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt  # if available
# or install manually as shown above
```

##  Dataset Sources

### Primary Datasets

#### 1. VoxPopuli Dataset
- **Source**: European Parliament speeches
- **Size**: 400+ hours of multilingual content
- **Language**: English subset (~200+ hours)
- **Format**: OGG compressed audio
- **Quality**: Professional recording quality
- **License**: CC0 (public domain)

#### 2. LibriSpeech Dataset
- **Source**: Audiobooks from LibriVox
- **Size**: Various subsets (dev-clean, test-clean, etc.)
- **Format**: FLAC compressed audio
- **Quality**: High-quality clean speech
- **License**: CC BY 4.0

### Supplementary Datasets (Optional)
- **Common Voice**: Mozilla's crowdsourced speech
- **GTZAN**: Music genres for robustness testing
- **FMA**: Free Music Archive for generalization

##  Quick Start

For users who want to get started immediately with a smaller dataset:

```bash


# Create required directories
mkdir -p data/{raw_audios/en,100,200,200_speech_only,test_5_hours}
mkdir -p dataset_creation classification_results

# Download VoxPopuli (English subset)
cd dataset_creation
python dataset.py

# Generate metadata for all files
python subset.py

# Create a small subset (5 hours for testing)
python select_audios.py metadata.csv --hours 5 test_set_5_hours_metadata.csv

# Process to 1-second segments
python 1_sec_files.py  # Edit script to use test_set_5_hours_metadata.csv

# Classify speech vs noise
python noise_mul.py --dir data/test_5_hours --output ./test_classification_results

# Extract speech-only files (edit speech_only.py configuration first)
# Configure speech_only.py:
# - Set speech_files_path = 'test_classification_results/speech_files.txt'
# - Set destination_folder = 'data/test_5_hours_speech_only'
python speech_only.py
```

##  Detailed Setup Instructions

### Step 1: Download Raw Datasets

#### Option A: VoxPopuli via Hugging Face (Recommended)

```bash
cd dataset_creation
python dataset.py
```

**What this does:**
- Downloads VoxPopuli English dataset via Hugging Face
- Caches dataset locally for reuse
- Extracts audio files to `data/raw_audios/en/`

**Expected output structure:**
```
data/raw_audios/en/
├── 20090112-0900-PLENARY-21_en.ogg
├── 20090113-0900-PLENARY-7_en.ogg
├── 20090114-0900-PLENARY-3_en.ogg
└── ... (hundreds more files)
```

#### Option B: Manual VoxPopuli Download

If automatic download fails, manually download from:
- **URL**: [VoxPopuli GitHub](https://github.com/facebookresearch/voxpopuli)
- **Direct**: [Hugging Face Dataset](https://huggingface.co/datasets/facebook/voxpopuli)

```bash
# Download specific language
# Follow VoxPopuli documentation for manual download
# Extract to data/raw_audios/en/
```

#### Option C: LibriSpeech Dataset

```bash
# Download LibriSpeech dev-clean subset
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf dev-clean.tar.gz
mv LibriSpeech/dev-clean data/dev-clean/
```

### Step 2: Generate Complete Metadata

```bash
cd dataset_creation
python subset.py
```

**Configuration (edit subset.py if needed):**
```python
root_directory = "data/raw_audios/en"  # Adjust path as needed
output_csv = "metadata.csv"
```

**Output:**
- `metadata.csv`: Complete inventory of all audio files with durations
- Format: `filepath,duration`

**Sample output:**
```csv
filepath,duration
data/raw_audios/en/20090112-0900-PLENARY-21_en.ogg,45.23
data/raw_audios/en/20090113-0900-PLENARY-7_en.ogg,189.45
...
```

### Step 3: Create Duration-Based Subsets

```bash
# Create 100-hour training subset
python select_audios.py metadata.csv --hours 100 100_hours_metadata.csv

# Create 200-hour training subset  
python select_audios.py metadata.csv --hours 200 200_hours_metadata.csv

# Create 5-hour test subset
python select_audios.py metadata.csv --hours 5 test_set_5_hours_metadata.csv
```

**Parameters:**
- `--hours X`: Select X hours of audio content
- `--seconds X`: Alternative selection by seconds
- Random shuffling ensures diverse content selection

### Step 4: Audio Preprocessing

#### Option A: Flat Structure (Recommended for large datasets)

```bash
python 1_sec_files.py
```

**Configuration (edit 1_sec_files.py):**
```python
subset_csv = "dataset_creation/200_hours_metadata.csv"  # Input metadata
output_directory = "data/200"                          # Output directory
max_workers = 8                                        # Adjust based on CPU cores
```

**For different datasets, modify these variables:**
- 100-hour dataset: `subset_csv = "dataset_creation/100_hours_metadata.csv"`, `output_directory = "data/100"`
- Test dataset: `subset_csv = "dataset_creation/test_set_5_hours_metadata.csv"`, `output_directory = "data/test_5_hours"`

**Processing features:**
- Converts OGG/FLAC to WAV format
- Resamples to 16kHz sample rate
- Normalizes amplitude to 0.99 maximum
- Creates 1-second segments with overlap
- Parallel processing for speed

#### Option B: Hierarchical Structure (For smaller datasets)

```bash
python 100_sub.py
```

**Configuration:**
```python
subset_csv = "dataset_creation/100_hours_metadata.csv"
output_directory = "data/100/preprocessed_audio"
max_workers = 4
```

**Differences from flat structure:**
- Creates subdirectories per original file
- Better organization for smaller datasets
- Easier to trace segments back to source files

#### Option C: Alternative Preprocessing (pre_100.py)

```bash
python pre_100.py
```

**Features:**
- Similar to 100_sub.py but with different output structure
- Creates individual directories per source file
- Handles resampling and normalization
- Segments audio into 1-second clips with file-based organization

### Step 5: Speech vs. Noise Classification

```bash
# Use optimized classification script
python noise_mul.py --dir data/200 --output ./classification_results --workers 16

# For smaller test runs with sampling
python noise_mul.py --dir data/200 --output ./classification_results --sample 10000 --workers 8
```

**Parameters:**
- `--dir`: Directory containing 1-second WAV files
- `--output`: Output directory for results
- `--workers`: Number of parallel workers
- `--sample`: Process only N random files (for testing)

**Output files:**
- `audio_classification_results.csv`: Complete analysis
- `speech_files.txt`: Files classified as speech
- `noise_files.txt`: Files classified as noise

**Classification features:**
- Energy distribution analysis
- Frequency band analysis (300-3000 Hz for speech)
- Zero-crossing rate computation
- Spectral features (centroid, bandwidth, rolloff)
- MFCC statistical analysis

### Step 6: Extract Speech-Only Dataset

```bash
python speech_only.py
```

**Configuration (edit speech_only.py):**
```python
speech_files_path = 'classification_results/speech_files.txt'
destination_folder = 'data/200_speech_only'
```

**For different datasets, modify these paths:**
- 100-hour dataset: `destination_folder = 'data/100_speech_only'`
- Test dataset: `speech_files_path = 'test_classification_results/speech_files.txt'`, `destination_folder = 'data/test_5_hours_speech_only'`

**What this does:**
- Copies files identified as speech to dedicated directory
- Filters out music, noise, and silence
- Creates clean training dataset

### Step 7: Quality Control (Optional)

```bash
# Detect silent files
python silent.py

# Check for dataset overlaps
python same.py
```

### File Naming Conventions

**Original files:**
- Format: `YYYYMMDD-HHMM-PLENARY-N_en.ogg`
- Example: `20090114-0900-PLENARY-3_en.ogg`

**Processed segments:**
- Format: `{original_name}_seg{number}.wav`
- Example: `20090114-0900-PLENARY-3_en_seg1.wav`

**Metadata structure:**
```csv
filepath,duration
data/raw_audios/en/20090112-0900-PLENARY-21_en.ogg,45.23
data/raw_audios/en/20090113-0900-PLENARY-7_en.ogg,189.45
```



### Integration with Training Scripts

#### Verify Dataset Compatibility
```python
# Test dataset loading
from py.main16 import OneSecClipsDataset

dataset = OneSecClipsDataset(root_dir="data/200_speech_only")
print(f"Dataset size: {len(dataset)} segments")

# Test single sample
sample = dataset[0]
print(f"Sample shape: {sample.shape}")  # Should be [1, 16000]
```

#### Dataset Statistics
```python
# Generate dataset statistics
import numpy as np
import torchaudio

def analyze_dataset(dataset_path):
    durations = []
    amplitudes = []
    
    for file in glob.glob(f"{dataset_path}/*.wav"):
        waveform, sr = torchaudio.load(file)
        durations.append(waveform.shape[1] / sr)
        amplitudes.append(waveform.abs().max().item())
    
    print(f"Average duration: {np.mean(durations):.3f}s")
    print(f"Max amplitude: {np.max(amplitudes):.3f}")
    print(f"Sample rate consistency: {sr} Hz")
```

##  Expected Results



### Quality Metrics
- **Segment duration consistency**: 1.000s ± 0.001s
- **Sample rate consistency**: 16,000 Hz exact
- **Amplitude normalization**: max ≤ 0.99

## Next Steps

After successful dataset creation:

1. **Verify data integrity**: Run sample checks on processed files
2. **Start training**: Use `python py/main16.py` with your processed data
3. **Monitor performance**: Track training metrics and data quality
4. **Iterate if needed**: Adjust classification thresholds or preprocessing parameters

