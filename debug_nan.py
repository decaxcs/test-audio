#!/usr/bin/env python3
"""
Debug script for NaN issues in StyleTTS2 second stage training
"""

import torch
import numpy as np
import yaml

def check_first_stage_checkpoint():
    """Check if first stage checkpoint has proper weights"""
    checkpoint_path = "Models/ThreeSpeaker/first_stage.pth"
    
    try:
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("=== First Stage Checkpoint Analysis ===")
        print(f"Keys in checkpoint: {list(state.keys())}")
        
        if 'net' in state:
            net_keys = list(state['net'].keys())
            print(f"\nModel components: {net_keys}")
            
            # Check for module prefix
            sample_keys = []
            for component in ['style_encoder', 'predictor_encoder', 'decoder']:
                if component in state['net']:
                    component_keys = list(state['net'][component].keys())[:3]
                    sample_keys.extend(component_keys)
                    print(f"\n{component} sample keys: {component_keys}")
            
            has_module_prefix = any('module.' in key for key in sample_keys)
            print(f"\nHas 'module.' prefix: {has_module_prefix}")
            
            # Check for NaN or extreme values
            for component, params in state['net'].items():
                if isinstance(params, dict):
                    for key, value in params.items():
                        if isinstance(value, torch.Tensor):
                            if torch.isnan(value).any():
                                print(f"WARNING: NaN found in {component}.{key}")
                            max_val = value.abs().max().item()
                            if max_val > 1000:
                                print(f"WARNING: Large values in {component}.{key}: max={max_val}")
        
        print("\n=== Checkpoint appears valid ===")
        return True
        
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return False

def check_config():
    """Verify configuration settings"""
    config_path = "Configs/config_multispeaker.yml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n=== Configuration Check ===")
    print(f"Batch size: {config['batch_size']} (MUST be >= 16)")
    print(f"Max length: {config['max_len']}")
    print(f"Learning rate: {config['optimizer_params']['lr']}")
    print(f"Multispeaker: {config['model_params']['multispeaker']}")
    print(f"Decoder type: {config['model_params']['decoder']['type']}")
    
    issues = []
    if config['batch_size'] < 16:
        issues.append("CRITICAL: batch_size < 16 will cause NaN losses!")
    
    if config['max_len'] > 300:
        issues.append("WARNING: max_len > 300 may cause OOM with batch_size 16")
    
    if issues:
        print("\n=== Issues Found ===")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\n=== Configuration looks good ===")
    
    return len(issues) == 0

def check_dataset():
    """Check dataset for issues"""
    train_list = "Data/train_list.txt"
    
    with open(train_list, 'r') as f:
        lines = f.readlines()
    
    print(f"\n=== Dataset Check ===")
    print(f"Total samples: {len(lines)}")
    
    # Check format
    issues = 0
    for i, line in enumerate(lines[:5]):
        parts = line.strip().split('|')
        if len(parts) != 3:
            print(f"ERROR: Line {i+1} has {len(parts)} parts, expected 3")
            issues += 1
    
    if issues == 0:
        print("Dataset format appears correct")
    
    # With 108 samples and batch_size 16, we have 6.75 batches
    print(f"\nWith batch_size 16: {len(lines)/16:.1f} batches per epoch")
    print("NOTE: Small dataset may require more epochs for convergence")
    
    return issues == 0

def main():
    """Run all checks"""
    print("StyleTTS2 NaN Debugging Tool")
    print("=" * 50)
    
    checks = [
        ("First Stage Checkpoint", check_first_stage_checkpoint),
        ("Configuration", check_config),
        ("Dataset", check_dataset)
    ]
    
    all_passed = True
    for name, check_func in checks:
        try:
            passed = check_func()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("All checks passed! Try running training again:")
        print("cd /root/StyleTTS2")
        print("python train_second.py --config_path ./Configs/config_multispeaker.yml")
    else:
        print("Issues found! Please fix them before training.")
    
    print("\nAdditional tips:")
    print("1. If still getting NaN, try reducing learning rates further")
    print("2. Monitor GPU memory - you may need to reduce max_len")
    print("3. Consider using gradient clipping if instability persists")

if __name__ == "__main__":
    main()