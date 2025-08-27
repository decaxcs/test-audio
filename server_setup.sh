#!/bin/bash
# StyleTTS2 Server Setup Script
# Run this on your server after git pull

echo "==================================="
echo "StyleTTS2 Server Setup Script"
echo "==================================="

# Check if we're in the right directory
if [ ! -f "train_first.py" ]; then
    echo "Error: Please run this script from the StyleTTS2 directory"
    exit 1
fi

# Fix PyTorch 2.6+ weights_only issue
echo "Fixing PyTorch loading issues..."
cat > fix_pytorch.py << 'EOF'
import os
import re

files_to_fix = ['models.py', 'train_first.py', 'train_second.py', 
                'Utils/PLBERT/util.py', 'train_finetune.py']

for filename in files_to_fix:
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            content = f.read()
        
        # Fix torch.load calls
        content = re.sub(
            r"torch\.load\(([^)]+)\)",
            lambda m: m.group(0) if 'weights_only' in m.group(0) 
                     else m.group(0).replace(')', ', weights_only=False)'),
            content
        )
        
        with open(filename, 'w') as f:
            f.write(content)
        print(f"Fixed {filename}")
EOF

python3 fix_pytorch.py
rm fix_pytorch.py

# Check for data files
echo ""
echo "Checking data files..."
if [ ! -d "Data/wavs" ]; then
    echo "Warning: Data/wavs directory not found!"
    echo "Please ensure your audio data is properly organized"
else
    echo "✓ Data directory found"
    ls -la Data/wavs/ | head -5
fi

if [ ! -f "Data/train_list.txt" ]; then
    echo "Warning: train_list.txt not found!"
else
    echo "✓ train_list.txt found ($(wc -l < Data/train_list.txt) samples)"
fi

if [ ! -f "Data/val_list.txt" ]; then
    echo "Warning: val_list.txt not found!"
else
    echo "✓ val_list.txt found ($(wc -l < Data/val_list.txt) samples)"
fi

# Check for config
echo ""
echo "Checking configuration..."
if [ -f "Configs/config_multispeaker.yml" ]; then
    echo "✓ config_multispeaker.yml found"
    grep -E "epochs_1st:|epochs_2nd:|batch_size:|max_len:" Configs/config_multispeaker.yml
else
    echo "Error: config_multispeaker.yml not found!"
fi

# Check GPU availability
echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "Warning: nvidia-smi not found. GPU may not be available."
fi

# Display training commands
echo ""
echo "==================================="
echo "Ready to train! Use these commands:"
echo "==================================="
echo ""
echo "1. First stage training (run first):"
echo "   accelerate launch --num_processes=1 train_first.py --config_path ./Configs/config_multispeaker.yml"
echo ""
echo "2. Monitor training (in separate terminal):"
echo "   tensorboard --logdir Models/ThreeSpeaker"
echo ""
echo "3. Second stage training (after first stage completes):"
echo "   python train_second.py --config_path ./Configs/config_multispeaker.yml"
echo ""
echo "==================================="
echo "Tips:"
echo "- Use 'screen' or 'tmux' to keep training running"
echo "- If OOM error, reduce batch_size in config (try 4 or 2)"
echo "- Training will take 2-4 days for both stages"
echo "==================================="