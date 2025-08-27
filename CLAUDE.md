# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StyleTTS2 is a state-of-the-art text-to-speech (TTS) synthesis system that uses style diffusion and adversarial training with large speech language models to achieve human-level TTS synthesis. It supports both single-speaker and multi-speaker models.

## Quick Start

After git pull, run the setup script:
```bash
bash server_setup.sh
```

## Commands

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# On Windows, also run:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -U

# For demo/inference:
pip install phonemizer
sudo apt-get install espeak-ng
```

### Training
```bash
# First stage training (uses accelerate with DDP)
accelerate launch train_first.py --config_path ./Configs/config.yml

# Second stage training (uses DP, DDP not working)
python train_second.py --config_path ./Configs/config.yml

# Fine-tuning on new speakers
python train_finetune.py --config_path ./Configs/config_ft.yml

# Fine-tuning with single GPU (faster, less VRAM)
accelerate launch --mixed_precision=fp16 --num_processes=1 train_finetune_accelerate.py --config_path ./Configs/config_ft.yml
```

### Testing
There are no explicit test commands in the codebase. The project uses Jupyter notebooks for inference demos in the `Demo/` and `Colab/` directories.

## Architecture

### Core Components

1. **Training Pipeline**
   - `train_first.py`: First stage pre-training with text aligner and pitch extractor
   - `train_second.py`: Second stage joint training with SLM adversarial training
   - `train_finetune.py` / `train_finetune_accelerate.py`: Fine-tuning on new speakers

2. **Model Architecture** (`models.py`)
   - **Text Encoder**: ProsodyPredictor with BERT-based text encoding
   - **Style Diffusion**: Diffusion-based style generation (Modules/diffusion/)
   - **Decoder**: Choice between HiFi-GAN or iSTFTNet
   - **Discriminators**: MultiPeriodDiscriminator, MultiResSpecDiscriminator, WavLMDiscriminator

3. **Pre-trained Components** (Utils/)
   - **ASR**: Text aligner pre-trained on LibriTTS/JVS/AiShell
   - **JDC**: Pitch extractor pre-trained on LibriTTS
   - **PLBERT**: Pre-trained language BERT for phoneme encoding

4. **Configuration System**
   - Config files in `Configs/` directory control all training parameters
   - Key configs: `config.yml` (LJSpeech), `config_libritts.yml` (LibriTTS), `config_ft.yml` (fine-tuning)

### Key Configuration Parameters
- `multispeaker`: Set to true for multi-speaker models
- `max_len`: Maximum audio length in frames (reduce if OOM)
- `batch_percentage`: Controls batch size for SLM training (reduce if OOM)
- `joint_epoch`: Controls when SLM adversarial training starts
- `OOD_data`: Path to out-of-distribution texts for adversarial training

### Data Format
Training data lists should follow: `filename.wav|transcription|speaker`

## Important Notes

1. **DDP Issue**: Second stage training (`train_second.py`) doesn't work with DDP, only DP is supported
2. **Mixed Precision**: Avoid for first stage training as it can cause NaN losses with small batch sizes
3. **Batch Size**: Recommended batch size is 16 to avoid NaN losses
4. **Multi-language Support**: Requires language-specific PL-BERT model
5. **GPU Requirements**: Modern GPUs recommended to avoid high-pitched noise artifacts