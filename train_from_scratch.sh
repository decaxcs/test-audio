#!/bin/bash
# Complete training script for StyleTTS2 with augmented data

echo "=================================="
echo "StyleTTS2 Training from Scratch"
echo "With Data Augmentation"
echo "=================================="

# Configuration
STYLETTS_DIR="/root/StyleTTS2"
CONFIG="./Configs/config_multispeaker_augmented.yml"

# Change to StyleTTS2 directory
cd $STYLETTS_DIR

# Step 1: Run data augmentation
echo -e "\n[Step 1] Running data augmentation..."
if [ -f "Data/train_list_augmented.txt" ]; then
    echo "Augmented data already exists. Checking..."
    ORIG_COUNT=$(wc -l < Data/train_list.txt)
    AUG_COUNT=$(wc -l < Data/train_list_augmented.txt)
    echo "Original samples: $ORIG_COUNT"
    echo "Augmented total: $AUG_COUNT"
    
    read -p "Regenerate augmented data? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python augment_data.py
    fi
else
    python augment_data.py
fi

# Step 2: Clean previous training
echo -e "\n[Step 2] Cleaning previous training artifacts..."
read -p "Backup existing checkpoints? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    BACKUP_DIR="Models/ThreeSpeaker_backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p $BACKUP_DIR
    echo "Backing up to $BACKUP_DIR..."
    mv Models/ThreeSpeaker/*.pth $BACKUP_DIR/ 2>/dev/null
    mv Models/ThreeSpeaker/train.log $BACKUP_DIR/ 2>/dev/null
    cp Models/ThreeSpeaker/config_multispeaker.yml $BACKUP_DIR/ 2>/dev/null
fi

# Clean tensorboard logs
rm -rf Models/ThreeSpeaker/tensorboard/*
echo "Cleaned tensorboard logs"

# Step 3: Setup environment
echo -e "\n[Step 3] Setting up environment..."
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb=128"
echo "Memory optimization enabled"

# Check GPU
echo -e "\nGPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

# Step 4: Start training
echo -e "\n[Step 4] Starting Stage 1 training..."
echo "Configuration:"
echo "- Config: $CONFIG"
echo "- Epochs: 400"
echo "- Batch size: 16"
echo "- Max length: 100"
echo "- Save frequency: every 20 epochs"
echo ""
echo "Expected training time: 16-24 hours"
echo "Target validation loss: 0.25-0.35"
echo ""

# Create log directory if not exists
mkdir -p Models/ThreeSpeaker

# Function to show monitoring commands
show_monitoring() {
    echo -e "\n=================================="
    echo "Training started! Monitor with:"
    echo ""
    echo "# Terminal 1 - Validation loss:"
    echo "tail -f Models/ThreeSpeaker/train.log | grep -E 'Validation loss:|Epochs:'"
    echo ""
    echo "# Terminal 2 - GPU usage:"
    echo "watch -n 2 nvidia-smi"
    echo ""
    echo "# Terminal 3 - TensorBoard:"
    echo "tensorboard --logdir Models/ThreeSpeaker/tensorboard --port 6006"
    echo "=================================="
}

# Start training with error handling
echo -e "\nPress Ctrl+C to cancel, starting in 5 seconds..."
sleep 5

# Show monitoring commands
show_monitoring

# Run training
accelerate launch train_first.py --config_path $CONFIG 2>&1 | tee Models/ThreeSpeaker/train_stage1.log

# Check if training completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "\n✅ Stage 1 training completed successfully!"
    
    # Find best checkpoint
    echo -e "\nFinding best checkpoint..."
    BEST_CHECKPOINT=$(grep "Validation loss:" Models/ThreeSpeaker/train.log | \
                      awk '{print $NF, NR}' | sort -n | head -1 | awk '{print $2}')
    echo "Best validation loss at line: $BEST_CHECKPOINT"
    
    # Prepare for Stage 2
    echo -e "\n[Step 5] Preparing for Stage 2..."
    read -p "Copy best checkpoint to first_stage.pth? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Find the checkpoint file
        LATEST=$(ls -t Models/ThreeSpeaker/epoch_1st_*.pth | head -1)
        cp $LATEST Models/ThreeSpeaker/first_stage.pth
        echo "Copied $LATEST to first_stage.pth"
        
        echo -e "\nReady for Stage 2! Run:"
        echo "python train_second_fixed.py --config_path $CONFIG"
    fi
else
    echo -e "\n❌ Training interrupted or failed!"
    echo "Check Models/ThreeSpeaker/train_stage1.log for errors"
    echo ""
    echo "To resume training, run:"
    echo "accelerate launch train_first.py --config_path $CONFIG"
fi

echo -e "\nTraining script completed!"