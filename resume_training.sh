#!/bin/bash
# Resume StyleTTS2 training with memory optimizations

echo "StyleTTS2 Training Resumption Script"
echo "===================================="

# Check for existing checkpoints
CHECKPOINT_DIR="Models/ThreeSpeaker"
LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/epoch_2nd_*.pth 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found. Starting fresh training..."
    RESUME_CMD=""
else
    echo "Found checkpoint: $LATEST_CHECKPOINT"
    # Extract epoch number from filename
    EPOCH=$(basename "$LATEST_CHECKPOINT" | sed 's/epoch_2nd_\([0-9]*\)\.pth/\1/')
    echo "Resuming from epoch: $((EPOCH + 1))"
    RESUME_CMD="--pretrained_model $LATEST_CHECKPOINT"
fi

echo ""
echo "Configuration:"
echo "- Max length: 100 (reduced from 150)"
echo "- Batch percentage: 0.2 (reduced from 0.3)"
echo "- Memory optimizations enabled"
echo ""

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb=128"

# Run training with optimizations
echo "Starting training with memory optimizations..."
python train_second_fixed.py \
    --config_path ./Configs/config_multispeaker.yml \
    $RESUME_CMD

echo "Training completed or interrupted."