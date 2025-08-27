# StyleTTS2 Server Training Guide

## Quick Start (After Git Pull)

```bash
# 1. Run setup script
bash server_setup.sh

# 2. Start first stage training
accelerate launch --num_processes=1 train_first.py --config_path ./Configs/config_multispeaker.yml

# 3. After first stage completes, run second stage
python train_second.py --config_path ./Configs/config_multispeaker.yml
```

## Your Dataset Info
- **3 speakers** (speaker_f, speaker_ff, speaker_m)
- **10 minutes per speaker** (30 minutes total)
- **108 training samples, 12 validation samples**

## Training Configuration
- **Stage 1:** 150 epochs (~1-2 days)
- **Stage 2:** 75 epochs (~1 day)
- **Batch size:** 8 (reduce if OOM)
- **Max length:** 200 frames

## Files Included
- `Configs/config_multispeaker.yml` - Optimized config for your dataset
- `server_setup.sh` - Auto-fixes PyTorch issues and checks setup
- `TROUBLESHOOTING.md` - Common errors and solutions
- `Data/` - Your audio data (already organized)

## Monitor Training
```bash
# In separate terminal
tensorboard --logdir Models/ThreeSpeaker
```

## If Training Crashes
Just run the same command again - it auto-resumes from checkpoint!

## Troubleshooting
See `TROUBLESHOOTING.md` for common issues.

## Expected Results
With only 30 minutes of data, expect:
- Lower quality than commercial TTS
- Some overfitting
- But should be usable for testing!

For production quality, collect 1+ hours per speaker.