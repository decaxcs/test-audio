# Multi-Speaker StyleTTS2 Training Guide

## Step 1: Data Preparation

### 1.1 Directory Structure
Create the following directory structure:
```
StyleTTS2/
├── Data/
│   ├── wavs/           # All audio files go here
│   │   ├── speaker1/
│   │   │   ├── audio1.wav
│   │   │   ├── audio2.wav
│   │   │   └── ...
│   │   ├── speaker2/
│   │   │   ├── audio1.wav
│   │   │   ├── audio2.wav
│   │   │   └── ...
│   │   └── speaker3/
│   │       └── ...
│   ├── train_list.txt  # Training data list
│   ├── val_list.txt     # Validation data list
│   └── OOD_texts.txt    # Out-of-distribution texts
```

### 1.2 Audio Requirements
- **Format**: WAV files
- **Sample Rate**: 24000 Hz (24 kHz)
- **Channels**: Mono
- **Duration**: Recommended 2-15 seconds per clip
- **Quality**: Clean recordings without background noise

### 1.3 Convert Audio to 24kHz
If your audio is not 24kHz, use this Python script to convert:

```python
import os
import librosa
import soundfile as sf
from glob import glob
from tqdm import tqdm

def convert_to_24khz(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    audio_files = glob(f"{input_dir}/**/*.wav", recursive=True)
    
    for audio_path in tqdm(audio_files):
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Resample to 24kHz if needed
        if sr != 24000:
            y = librosa.resample(y, orig_sr=sr, target_sr=24000)
        
        # Maintain directory structure
        rel_path = os.path.relpath(audio_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as 24kHz wav
        sf.write(output_path, y, 24000)
        
    print(f"Converted {len(audio_files)} files to 24kHz")

# Usage
convert_to_24khz("path/to/original/audio", "Data/wavs")
```

### 1.4 Create Data Lists
The data lists must follow this format:
```
path/to/audio.wav|transcription text|speaker_name
```

Example Python script to create data lists:

```python
import os
import random
from glob import glob

def create_data_lists(wav_dir, transcriptions_dict, output_dir="Data", val_split=0.05):
    """
    wav_dir: Path to directory containing speaker folders with wav files
    transcriptions_dict: Dictionary mapping audio_path to transcription text
    val_split: Percentage of data to use for validation
    """
    
    all_data = []
    
    # Iterate through each speaker directory
    for speaker_dir in os.listdir(wav_dir):
        speaker_path = os.path.join(wav_dir, speaker_dir)
        
        if not os.path.isdir(speaker_path):
            continue
            
        speaker_name = speaker_dir
        
        # Get all wav files for this speaker
        wav_files = glob(os.path.join(speaker_path, "*.wav"))
        
        for wav_file in wav_files:
            # Get relative path from wav_dir
            rel_path = os.path.relpath(wav_file, wav_dir)
            
            # Get transcription (you need to provide this mapping)
            transcription = transcriptions_dict.get(rel_path, "")
            
            if transcription:
                # Format: path|text|speaker
                all_data.append(f"{rel_path}|{transcription}|{speaker_name}")
    
    # Shuffle and split
    random.shuffle(all_data)
    split_idx = int(len(all_data) * (1 - val_split))
    
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    # Write to files
    with open(os.path.join(output_dir, "train_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_data))
    
    with open(os.path.join(output_dir, "val_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_data))
    
    print(f"Created train_list.txt with {len(train_data)} samples")
    print(f"Created val_list.txt with {len(val_data)} samples")
    print(f"Total speakers: {len(set([line.split('|')[2] for line in all_data]))}")

# Example usage with a CSV file containing transcriptions
import pandas as pd

# If you have a CSV with columns: filename, text
df = pd.read_csv("transcriptions.csv")
transcriptions = dict(zip(df['filename'], df['text']))

create_data_lists("Data/wavs", transcriptions)
```

### 1.5 Create OOD (Out-of-Distribution) Texts
Create `Data/OOD_texts.txt` with random texts not in your training data:

```python
# Example OOD texts - these should be different from your training texts
ood_texts = [
    "The weather forecast predicts rain for tomorrow afternoon.|dummy",
    "Scientists discovered a new species of butterfly in the Amazon.|dummy",
    "The stock market showed significant gains this quarter.|dummy",
    # Add at least 100-500 diverse sentences
]

with open("Data/OOD_texts.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(ood_texts))
```

## Step 2: Configuration

### 2.1 Copy and Modify Config File
```bash
cp Configs/config_libritts.yml Configs/config_multispeaker.yml
```

### 2.2 Edit Configuration
Edit `Configs/config_multispeaker.yml`:

```yaml
log_dir: "Models/MultiSpeaker"
first_stage_path: "first_stage.pth"
save_freq: 5
log_interval: 10
device: "cuda"
epochs_1st: 200  # Adjust based on dataset size
epochs_2nd: 100  # Adjust based on dataset size
batch_size: 16   # Reduce if OOM
max_len: 400     # Reduce if OOM (400 = 5 seconds at 24kHz)

data_params:
  train_data: "Data/train_list.txt"
  val_data: "Data/val_list.txt"
  root_path: "Data/wavs"  # Root directory for audio files
  OOD_data: "Data/OOD_texts.txt"
  min_length: 50

model_params:
  multispeaker: true  # IMPORTANT: Must be true for multi-speaker

# Keep other parameters as default unless you know what you're doing
```

## Step 3: Training Pipeline

### 3.1 Install Dependencies
```bash
# Install requirements
pip install -r requirements.txt

# Install phonemizer (required for text processing)
pip install phonemizer
sudo apt-get install espeak-ng

# For GPU support (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3.2 First Stage Training (Pre-training)
```bash
# This trains the core model components
accelerate launch train_first.py --config_path ./Configs/config_multispeaker.yml
```
- This will take approximately 1-3 days depending on dataset size and GPU
- Checkpoints saved as: `Models/MultiSpeaker/epoch_1st_00XXX.pth`
- Monitor tensorboard: `tensorboard --logdir Models/MultiSpeaker`

### 3.3 Second Stage Training (Joint Training with Adversarial)
```bash
# This adds adversarial training for better quality
python train_second.py --config_path ./Configs/config_multispeaker.yml
```
- This will take approximately 1-2 days
- Checkpoints saved as: `Models/MultiSpeaker/epoch_2nd_00XXX.pth`
- Note: This uses DataParallel (DP) not DistributedDataParallel (DDP)

### 3.4 Monitor Training
```bash
# In a separate terminal, run:
tensorboard --logdir Models/MultiSpeaker
# Then open http://localhost:6006 in your browser
```

## Step 4: Common Issues and Solutions

### Memory Issues (OOM)
```yaml
# In config file, reduce these parameters:
batch_size: 8      # Try 8 or even 4
max_len: 300       # Reduce to 300 or 200
batch_percentage: 0.5  # For second stage training
```

### NaN Loss
- Don't use mixed precision for first stage
- Ensure batch_size >= 16 (or use gradient accumulation)
- Check your audio files for corruption or silence

### Data Requirements
- **Minimum**: 10 hours of audio per speaker, 3+ speakers
- **Recommended**: 20+ hours per speaker, 5+ speakers
- **Optimal**: 50+ hours per speaker, 10+ speakers

### Training Time Estimates (on 4x A100 GPUs)
- First Stage: ~2 days for 100 hours of data
- Second Stage: ~1 day for 100 hours of data

## Step 5: Inference with Trained Model

Create an inference script:

```python
import torch
import yaml
from models import *
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DPMSolver
import numpy as np

# Load config
config = yaml.safe_load(open("Configs/config_multispeaker.yml"))

# Load models
model = build_model(config['model_params'])
model.load_state_dict(torch.load("Models/MultiSpeaker/epoch_2nd_00100.pth"))
model.eval()

# Synthesize speech
text = "Hello, this is a test of multi-speaker synthesis."
speaker_id = "speaker1"  # Use actual speaker name from training

# Generate audio (implement based on Demo notebooks)
audio = model.synthesize(text, speaker_id)
```

## Step 6: Fine-tuning for New Speakers

Once you have a base model, you can fine-tune for new speakers:

```bash
# Prepare 1-2 hours of new speaker data
# Update config_ft.yml with new speaker data
python train_finetune.py --config_path ./Configs/config_ft.yml
```

## Important Notes

1. **Data Quality > Quantity**: Clean, well-recorded audio is crucial
2. **Speaker Diversity**: More diverse speakers = better generalization
3. **Balanced Data**: Try to have similar amounts of data per speaker
4. **Transcription Accuracy**: Ensure transcriptions match audio exactly
5. **Regular Checkpointing**: Models are saved every `save_freq` epochs

## Quick Start Checklist

- [ ] Prepare audio files in 24kHz mono WAV format
- [ ] Organize audio into speaker folders
- [ ] Create accurate transcriptions for all audio
- [ ] Generate train_list.txt and val_list.txt
- [ ] Create OOD_texts.txt with diverse texts
- [ ] Copy and configure config_multispeaker.yml
- [ ] Install all dependencies
- [ ] Run first stage training
- [ ] Run second stage training
- [ ] Test inference with trained model