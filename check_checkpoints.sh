#!/bin/bash
# Script to check checkpoint status on server

echo "StyleTTS2 Checkpoint Status"
echo "=========================="

cd /root/StyleTTS2

# Check latest first stage checkpoint
echo -e "\nLatest First Stage Checkpoints:"
ls -lht Models/ThreeSpeaker/epoch_1st_*.pth | head -5

# Check first_stage.pth
echo -e "\nChecking first_stage.pth:"
if [ -f "Models/ThreeSpeaker/first_stage.pth" ]; then
    ls -lh Models/ThreeSpeaker/first_stage.pth
    echo "Note: This is typically the checkpoint used for stage 2 training"
else
    echo "first_stage.pth not found"
fi

# Check training log for last recorded validation loss
echo -e "\nLast 5 Validation Loss entries from training log:"
grep "Validation loss:" Models/ThreeSpeaker/train.log | tail -5

# Check if epoch 150 was completed
echo -e "\nChecking if epoch 150 was completed:"
grep "Epochs: 150" Models/ThreeSpeaker/train.log | tail -1

# Suggest next steps
echo -e "\n=========================="
echo "Next Steps:"
echo "1. If epoch 150 completed but didn't save, you can:"
echo "   - Continue from epoch 140 with reduced learning rate"
echo "   - The 0.572 validation loss at epoch 150 shows good progress"
echo ""
echo "2. To continue training from epoch 140:"
echo "   accelerate launch train_first.py --config_path ./Configs/config_multispeaker_continue.yml"
echo ""
echo "3. Or manually create first_stage.pth from epoch 140:"
echo "   cp Models/ThreeSpeaker/epoch_1st_00140.pth Models/ThreeSpeaker/first_stage.pth"