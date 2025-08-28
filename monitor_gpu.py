#!/usr/bin/env python3
"""
Monitor GPU memory usage during StyleTTS2 training
"""

import torch
import time
import subprocess
import sys

def get_gpu_memory():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(0) / 1024**3    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        free = total - allocated
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'free': free,
            'percentage': (allocated / total) * 100
        }
    return None

def monitor_training():
    """Monitor GPU memory during training"""
    print("GPU Memory Monitor for StyleTTS2")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("No CUDA GPU available!")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("\nMonitoring... (Press Ctrl+C to stop)")
    print("\nTime     | Allocated | Reserved | Free    | Usage")
    print("-" * 50)
    
    try:
        while True:
            mem = get_gpu_memory()
            timestamp = time.strftime("%H:%M:%S")
            print(f"{timestamp} | {mem['allocated']:8.2f} | {mem['reserved']:8.2f} | {mem['free']:7.2f} | {mem['percentage']:5.1f}%", end='\r')
            
            # Alert if memory usage is critical
            if mem['percentage'] > 95:
                print("\n⚠️  WARNING: GPU memory usage critical!")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

def check_training_config():
    """Check current training configuration"""
    print("\n\nCurrent Configuration Check:")
    print("-" * 30)
    
    config_file = "Configs/config_multispeaker.yml"
    important_params = ['batch_size', 'max_len', 'batch_percentage', 'diff_epoch', 'joint_epoch']
    
    try:
        with open(config_file, 'r') as f:
            for line in f:
                for param in important_params:
                    if param in line and ':' in line:
                        print(line.strip())
    except FileNotFoundError:
        print(f"Config file {config_file} not found")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check_training_config()
    else:
        monitor_training()