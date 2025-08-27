# StyleTTS2 Troubleshooting Guide

## Common Errors and Solutions

### 1. PyTorch weights_only Error
**Error:** `_pickle.UnpicklingError: Weights only load failed`

**Solution:** Run the server setup script:
```bash
bash server_setup.sh
```

Or manually fix by downgrading PyTorch:
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

### 2. KeyError: 'optimizer_params'
**Error:** `KeyError: 'optimizer_params'`

**Solution:** Your config file has wrong format. Use the provided `config_multispeaker.yml`

### 3. Out of Memory (OOM)
**Error:** `CUDA out of memory`

**Solutions:**
1. Reduce batch_size in config:
   ```yaml
   batch_size: 4  # or even 2
   ```

2. Reduce max_len:
   ```yaml
   max_len: 100  # from 200
   ```

3. Use gradient checkpointing (add to train command):
   ```bash
   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 accelerate launch ...
   ```

### 4. Multiple GPUs Detected
**Error:** `More than one GPU was found, enabling multi-GPU training`

**Solution:** Force single GPU:
```bash
accelerate launch --num_processes=1 train_first.py --config_path ./Configs/config_multispeaker.yml
```

Or set CUDA device:
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch ...
```

### 5. Data Not Found
**Error:** `FileNotFoundError: Data/train_list.txt`

**Solution:** Ensure data is properly organized:
```bash
ls -la Data/
# Should show: train_list.txt, val_list.txt, OOD_texts.txt, wavs/
```

### 6. Audio Sample Rate Wrong
**Error:** Related to audio processing or unexpected tensor shapes

**Solution:** Ensure all audio is 24kHz mono:
```bash
ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate,channels Data/wavs/*/*.wav
```

### 7. Training Crashes After Few Epochs
**Possible Causes:**
- NaN loss (batch size too small)
- Gradient explosion
- Corrupted audio files

**Solutions:**
1. Ensure batch_size >= 4 (or use gradient accumulation)
2. Check for silence or corrupted audio files
3. Reduce learning rate:
   ```yaml
   optimizer_params:
     lr: 5e-5  # from 1e-4
   ```

### 8. Very Slow Training
**Solutions:**
1. Check GPU utilization:
   ```bash
   watch -n 1 nvidia-smi
   ```

2. Reduce logging frequency:
   ```yaml
   log_interval: 100  # from 10
   save_freq: 50  # from 25
   ```

3. Use mixed precision (if stable):
   ```bash
   accelerate launch --mixed_precision=fp16 ...
   ```

### 9. Poor Quality Output
**Expected with small dataset!** But you can try:
1. Train longer (increase epochs)
2. Collect more data (target 1+ hour per speaker)
3. Fine-tune from pre-trained model instead
4. Reduce model complexity for small data

### 10. Connection Drops During Training
**Solution:** Use screen or tmux:
```bash
screen -S training
# Run training command
# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

## Monitoring Training

### Check Training Progress
```bash
# View latest log
tail -f Models/ThreeSpeaker/train.log

# Check checkpoint files
ls -lah Models/ThreeSpeaker/*.pth

# Monitor GPU usage
nvidia-smi -l 1
```

### TensorBoard
```bash
tensorboard --logdir Models/ThreeSpeaker --port 6006
# Access at http://localhost:6006
```

## Emergency Recovery

### Resume from Checkpoint
Training auto-resumes from latest checkpoint. Just run the same command again.

### Backup Important Files
```bash
# Backup checkpoints regularly
cp -r Models/ThreeSpeaker /backup/location/
```

### Reset and Start Fresh
```bash
rm -rf Models/ThreeSpeaker
mkdir -p Models/ThreeSpeaker
# Then start training again
```

## Performance Tips

1. **For 8GB VRAM:**
   - batch_size: 2
   - max_len: 100
   - Consider CPU training (very slow)

2. **For 16GB VRAM:**
   - batch_size: 4-8
   - max_len: 200

3. **For 24GB+ VRAM:**
   - batch_size: 16
   - max_len: 300-400

## Still Having Issues?

1. Check GitHub issues: https://github.com/yl4579/StyleTTS2/issues
2. Verify your environment:
   ```bash
   python --version  # Should be 3.8+
   pip list | grep torch  # Check PyTorch version
   nvidia-smi  # Check GPU
   ```
3. Try the Docker container if available
4. Consider using Google Colab for testing