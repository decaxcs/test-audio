# Complete Guide: Training StyleTTS2 from Scratch with Augmented Data

## Step 1: Data Augmentation (FIRST!)

### 1.1 Run augmentation script:
```bash
cd /root/StyleTTS2
python augment_data.py
```

This will create:
- ~1,500+ augmented samples from your 108 originals
- `Data/train_list_augmented.txt` - training list with all samples
- `Data/val_list_augmented.txt` - validation list with mild augmentations

### 1.2 Verify augmentation:
```bash
# Check augmented file count
wc -l Data/train_list_augmented.txt
# Should show ~1600+ lines (108 original + augmentations)

# Verify audio files created
ls Data/wavs/*aug*.wav | wc -l
```

## Step 2: Optimized Configuration

### 2.1 Create fresh training config:
Use `Configs/config_multispeaker_augmented.yml` (created below)

Key optimizations:
- Extended epochs (400 for stage 1)
- Gradual learning rate warmup
- Optimized for augmented dataset

## Step 3: Clean Previous Training

### 3.1 Backup old checkpoints:
```bash
# Create backup directory
mkdir -p Models/ThreeSpeaker_backup
mv Models/ThreeSpeaker/* Models/ThreeSpeaker_backup/
```

### 3.2 Clean directories:
```bash
# Remove old logs
rm -rf Models/ThreeSpeaker/tensorboard/*
rm -f Models/ThreeSpeaker/train.log
```

## Step 4: Start Fresh Training

### 4.1 Stage 1 Training:
```bash
cd /root/StyleTTS2

# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb=128"

# Start training with augmented data
accelerate launch train_first.py --config_path ./Configs/config_multispeaker_augmented.yml
```

### 4.2 Monitor training:
```bash
# Terminal 1 - Watch validation loss
tail -f Models/ThreeSpeaker/train.log | grep -E "Validation loss:|Epochs:"

# Terminal 2 - Monitor GPU
watch -n 2 nvidia-smi

# Terminal 3 - Check checkpoints
watch -n 60 'ls -lht Models/ThreeSpeaker/epoch_1st_*.pth | head -5'
```

## Step 5: Expected Training Timeline

With augmented data (~1,600 samples):

### Stage 1 (Text Alignment):
- **Epochs 1-50**: Rapid improvement (1.0+ → ~0.6)
- **Epochs 50-150**: Steady progress (0.6 → ~0.4)
- **Epochs 150-300**: Fine-tuning (0.4 → ~0.3)
- **Epochs 300-400**: Convergence (0.3 → ~0.25-0.3)

### Validation Loss Milestones:
- **Hour 1-2**: Should drop below 0.8
- **Hour 4-6**: Should reach ~0.5
- **Hour 8-12**: Should approach 0.3-0.4
- **Hour 16-24**: Final convergence

## Step 6: Stage 2 Training

After Stage 1 completes:

### 6.1 Create first_stage.pth:
```bash
# Use best checkpoint (lowest validation loss)
cp Models/ThreeSpeaker/epoch_1st_003XX.pth Models/ThreeSpeaker/first_stage.pth
```

### 6.2 Start Stage 2:
```bash
python train_second_fixed.py --config_path ./Configs/config_multispeaker_augmented.yml
```

## Step 7: Training Best Practices

### 7.1 Early Stopping:
- If validation loss increases for 30+ epochs, stop
- If validation loss plateaus for 50+ epochs, consider stopping

### 7.2 Learning Rate Adjustment:
- If loss plateaus early, reduce learning rate by half
- If loss spikes, reduce learning rate by 10x

### 7.3 Checkpoint Selection:
- Always use checkpoint with lowest validation loss
- Not necessarily the latest checkpoint

## Step 8: Troubleshooting

### If validation loss stays high (>0.8):
1. Check augmented audio files are valid
2. Verify transcriptions match audio
3. Reduce learning rate

### If training crashes:
1. Reduce batch_size to 8 (though may affect quality)
2. Reduce max_len to 80
3. Check GPU memory with `nvidia-smi`

### If loss becomes NaN:
1. Ensure batch_size >= 16
2. Check for corrupted audio files
3. Disable mixed precision for stage 1

## Step 9: Quality Expectations

With 108 original samples + augmentation:
- **Good TTS**: Validation loss 0.3-0.4
- **Very Good TTS**: Validation loss 0.25-0.3
- **Excellent TTS**: Validation loss < 0.25

Note: These are rough guidelines. Listen to synthesized samples for true quality assessment.

## Quick Reference Commands

```bash
# 1. Augment data
python augment_data.py

# 2. Clean and start fresh
rm -rf Models/ThreeSpeaker/tensorboard/* Models/ThreeSpeaker/*.pth
accelerate launch train_first.py --config_path ./Configs/config_multispeaker_augmented.yml

# 3. Monitor
tail -f Models/ThreeSpeaker/train.log | grep "Validation loss:"

# 4. Resume if interrupted
accelerate launch train_first.py --config_path ./Configs/config_multispeaker_augmented.yml
```