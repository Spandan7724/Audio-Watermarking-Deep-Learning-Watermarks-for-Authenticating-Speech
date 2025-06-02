# Audio Watermarking: Deep Learning Solution for Audio Authentication

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive deep learning solution for audio watermarking that provides authentication against audio deepfakes. This system uses a generator-detector architecture to embed imperceptible watermarks in audio files and reliably detect them, enabling validation of authentic audio content.

##  Overview

With the rise of sophisticated audio deepfakes and synthetic voice cloning, there's an urgent need for robust audio authentication methods. This project implements a learned watermarking system that:

- **Embeds imperceptible watermarks** in audio signals below psychoacoustic thresholds
- **Maintains audio quality** with negligible perceptual impact
- **Survives common transformations** like compression, resampling, and volume changes
- **Enables fast training** on consumer GPUs using 1-second audio clips
- **Provides reliable detection** with configurable thresholds

##  Architecture

### Generator Network
The generator employs an encoder-decoder architecture with LSTM bottleneck:
- **Encoder**: CNN layers with residual blocks for feature extraction
- **LSTM**: Temporal modeling for audio sequences  
- **Message Embedding**: Optional bit string encoding for payload
- **Decoder**: Transposed convolutions to generate watermark delta

### Detector Network  
The detector uses CNN-based architecture for sample-level detection:
- **Feature Extraction**: Residual CNN blocks for audio analysis
- **Sample-level Prediction**: Per-sample watermark probability
- **Message Decoding**: Optional bit string recovery from watermark

### Loss Functions
Multi-objective training with:
- **L1 Loss**: Minimize watermark energy
- **Multi-scale Mel Loss**: Preserve perceptual audio quality
- **Loudness Loss**: Maintain psychoacoustic masking
- **Detection Loss**: Optimize watermark detectability
- **High-frequency Penalty**: Limit artifacts in sensitive frequency ranges

##  Datasets

The project supports multiple audio datasets:

### Primary Training Data
- **VoxPopuli Dataset**: 200+ hours of European Parliament speeches https://github.com/facebookresearch/voxpopuli
- **LibriSpeech**: Clean speech recordings for validation https://www.openslr.org/12
- **Custom Preprocessing**: 1-second segments at 16kHz sample rate

### Dataset Organization
```
data/
├── 100/                    # 100-hour subset
├── 200/                    # 200-hour full dataset  
├── speech_only/            # Speech-filtered data
├── test_5_hours/           # Test set
└── raw_audios/             # Original recordings
```

### Data Processing Pipeline
1. **Audio Segmentation**: Convert long recordings to 1-second clips
2. **Quality Filtering**: Remove silent and noisy segments
3. **Speech Classification**: Separate speech from noise using acoustic features
4. **Normalization**: Standardize sample rate and amplitude

##  Quick Start


``` The models directory contains the trained models ```


### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/audio-watermarking.git
cd audio-watermarking

# Install dependencies
pip install torch torchaudio matplotlib tqdm numpy librosa soundfile pandas scipy
```

### Basic Usage

#### 1. Generate Watermarked Audio
```python
from py.main16 import generate_watermarked_audio, Generator
import torch

# Load pre-trained generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(message_bits=16)
generator.load_state_dict(torch.load('models/generator_best.pth'))
generator.eval()

# Generate watermarked audio
generate_watermarked_audio(
    input_file="your_audio.wav",
    generator=generator,
    output_file="watermarked_audio.wav",
    message_bits=16,
    device=device
)
```

#### 2. Detect Watermarks
```python
from py.main16 import detect_watermark, Detector

# Load pre-trained detector
detector = Detector(message_bits=16)
detector.load_state_dict(torch.load('models/detector_best.pth'))
detector.eval()

# Detect watermark
detection_result = detect_watermark(
    input_file="watermarked_audio.wav",
    detector=detector,
    detection_threshold=0.5,
    visualize=True,
    device=device
)
```

##  Training Variants

The project includes several training configurations optimized for different use cases:

| Script | Purpose | Key Features |
|--------|---------|--------------|
| `main14.py` | **Baseline with diagnostics** | Original architecture (2-block CNN → LSTM → CNN) and standard loss weights. Computes PESQ, STOI, SI-SNR and plots confusion matrices after training. |
| `main14b_2.py` | **Architecture exploration** |  Replaces the shallow encoder with a configurable residual stack (`HIDDEN_DIM`, `STRIDES`) and a 2-layer LSTM bottleneck. Ideal for architecture ablations. |
| `main14d.py` | **Extended training** | Extends the One-Cycle-LR schedule to **20 epochs**, keeps mid-epoch quick-validation and writes loss-curve PNGs. |
| `main15.py` | **Lightweight baseline** |  Same model as `main14` but drops heavy external metrics; useful on machines without PESQ/STOI dependencies. |
| `main16.py` | **Audio-quality focused** | Adds differentiable **RMS cap**, **peak clamp** and a **high-frequency penalty** to keep δ under –40 dBFS and below 3.5 kHz; λ-weights tightened accordingly. |








### Key hyperparameters:

```python
SAMPLE_RATE = 16000        # Audio sample rate
AUDIO_LEN = 16000          # 1-second clips
BATCH_SIZE = 16            # Training batch size
MESSAGE_BITS = 16          # Watermark payload size
EPOCHS = 10                # Training epochs
LR = 1e-3                  # Learning rate

# Loss weights
LAMBDA_L1 = 1.0            # Watermark energy penalty
LAMBDA_MSSPEC = 4.0        # Mel-spectrogram preservation
LAMBDA_LOUD = 20.0         # Loudness masking
LAMBDA_LOC = 10.0          # Detection accuracy
LAMBDA_DEC = 1.0           # Message decoding accuracy
```

##  Utilities and Scripts

### Dataset Creation
- `dataset_creation/1_sec_files.py`: Convert long audio files to 1-second segments
- `dataset_creation/select_audios.py`: Filter and curate audio datasets
- `dataset_creation/subset.py`: Create training/validation splits

### Audio Analysis
- `noise.py`: Classify audio as speech vs. noise using acoustic features
- `silent.py`: Detect and remove silent audio segments
- `speech_only.py`: Filter datasets to speech-only content

### Preprocessing
- Automatic resampling to 16kHz
- Amplitude normalization
- Stereo to mono conversion
- Padding/truncation to fixed length

##  Evaluation Metrics

### Audio Quality
- **PESQ**: Perceptual evaluation of speech quality
- **STOI**: Short-time objective intelligibility
- **SI-SNR**: Scale-invariant signal-to-noise ratio
- **Mel-spectrogram Distance**: Frequency domain similarity

### Detection Performance
- **True Positive Rate**: Correctly detected watermarks
- **False Positive Rate**: False alarms on clean audio
- **Bit Error Rate**: Message decoding accuracy
- **ROC Curves**: Threshold optimization

### Robustness Testing
- **Compression**: MP3, AAC encoding at various bitrates
- **Resampling**: Sample rate conversion artifacts
- **Volume Changes**: Amplitude scaling robustness
- **Additive Noise**: White noise and realistic interference

##  Advanced Features

### Message Embedding
Embed custom bit strings in watermarks:
```python
message = torch.randint(0, 2**16, (batch_size,))  # 16-bit messages
watermark = generator(audio, message=message)
```

### Adaptive Thresholding
Dynamic detection thresholds based on audio content:
```python
detection_prob = detect_prob(audio_file, detector)
adaptive_threshold = compute_adaptive_threshold(audio_file)
is_watermarked = detection_prob > adaptive_threshold
```

### Batch Processing
Process multiple files efficiently:
```python
process_folder_with_tqdm(
    input_folder="audio_folder/",
    generator=generator,
    output_folder="watermarked_folder/"
)
```


##  Web Application

For easy deployment and user interaction, we've developed a complete web application stack:

### Frontend: Next.js Application
- **Modern React Framework**: Built with Next.js for optimal performance
- **User-friendly Interface**: Intuitive audio upload and processing workflow
- **Real-time Feedback**: Progress indicators and result visualization
- **Responsive Design**: Works across desktop and mobile devices

### Backend: FastAPI with Uvicorn
- **High-performance API**: FastAPI framework for async audio processing
- **Model Integration**: Direct integration with PyTorch generator/detector models
- **File Upload Handling**: Efficient audio file processing pipeline
- **RESTful Endpoints**: Clean API design for watermark generation and detection

### Deployment Ready
- **Docker Support**: Containerized deployment for easy scaling
- **Production Optimized**: Configured for cloud deployment (AWS, GCP, Azure)
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation

## **🔗 Web Application Repository**: [Audio Watermarking Web App](https://github.com/Spandan7724/Audio-Watermarking-System)

*The web application provides a complete end-to-end solution for non-technical users to generate and detect audio watermarks through an intuitive web interface.*

##  Research Applications

This watermarking system enables research in:

- **Deepfake Detection**: Proactive protection against synthetic audio
- **Audio Forensics**: Chain of custody for audio evidence
- **Broadcast Authentication**: Verify news and media content
- **Content Protection**: Copyright and ownership verification
- **AI Safety**: Responsible AI development and deployment

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- VoxPopuli dataset creators for multilingual speech data
- LibriSpeech project for clean speech recordings
- PyTorch team for the deep learning framework

---