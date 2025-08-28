#!/usr/bin/env python3
"""
Inspect the exact location where NaN first appears in StyleTTS2
"""

import torch
import torch.nn as nn
import numpy as np
from models import build_model
import yaml
from munch import Munch

def trace_nan_source():
    """Trace where NaN originates in the model"""
    
    # Load config
    config = yaml.safe_load(open('Configs/config_multispeaker.yml'))
    
    # Build model
    model_params = Munch(config['model_params'])
    model = build_model(model_params)
    
    print("=== Model Architecture ===")
    for key, module in model.items():
        print(f"{key}: {type(module).__name__}")
    
    # Load checkpoint
    checkpoint_path = "Models/ThreeSpeaker/first_stage.pth"
    try:
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"\n=== Checkpoint keys: {list(state.keys())} ===")
        
        if 'net' in state:
            for key in state['net'].keys():
                print(f"Component in checkpoint: {key}")
                
            # Check for NaN in checkpoint
            print("\n=== Checking checkpoint for NaN ===")
            for component, params in state['net'].items():
                if isinstance(params, dict):
                    nan_count = 0
                    extreme_count = 0
                    for param_name, param_value in params.items():
                        if isinstance(param_value, torch.Tensor):
                            if torch.isnan(param_value).any():
                                nan_count += 1
                                print(f"NaN found in {component}.{param_name}")
                            if param_value.abs().max() > 1e6:
                                extreme_count += 1
                                print(f"Extreme values in {component}.{param_name}: max={param_value.abs().max():.2e}")
                    
                    if nan_count == 0 and extreme_count == 0:
                        print(f"{component}: OK")
                    else:
                        print(f"{component}: {nan_count} NaN, {extreme_count} extreme")
                        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Create dummy inputs
    print("\n=== Testing forward pass with dummy data ===")
    batch_size = 16
    device = 'cpu'
    
    # Create dummy inputs matching your data
    dummy_texts = torch.randint(0, 100, (batch_size, 50)).to(device)
    dummy_input_lengths = torch.tensor([50] * batch_size).to(device)
    dummy_mels = torch.randn(batch_size, 80, 200).to(device)  # 80 mel bins, 200 frames
    dummy_mel_lengths = torch.tensor([200] * batch_size).to(device)
    
    # Test style encoder
    if 'style_encoder' in model:
        print("\nTesting style_encoder...")
        try:
            with torch.no_grad():
                style_out = model['style_encoder'](dummy_mels.unsqueeze(1))  # Add channel dim
                print(f"Style encoder output: {style_out.shape}, has NaN: {torch.isnan(style_out).any()}")
                print(f"Style encoder stats: min={style_out.min():.3e}, max={style_out.max():.3e}, mean={style_out.mean():.3e}")
        except Exception as e:
            print(f"Style encoder error: {e}")
    
    # Test predictor encoder  
    if 'predictor_encoder' in model:
        print("\nTesting predictor_encoder...")
        try:
            with torch.no_grad():
                pred_out = model['predictor_encoder'](dummy_mels.unsqueeze(1))
                print(f"Predictor encoder output: {pred_out.shape}, has NaN: {torch.isnan(pred_out).any()}")
                print(f"Predictor encoder stats: min={pred_out.min():.3e}, max={pred_out.max():.3e}, mean={pred_out.mean():.3e}")
        except Exception as e:
            print(f"Predictor encoder error: {e}")
    
    # Check if encoder weights match
    if 'style_encoder' in model and 'predictor_encoder' in model:
        print("\n=== Checking encoder weight matching ===")
        style_params = dict(model['style_encoder'].named_parameters())
        pred_params = dict(model['predictor_encoder'].named_parameters())
        
        mismatch_count = 0
        for name in list(style_params.keys())[:5]:  # Check first 5 params
            if name in pred_params:
                match = torch.equal(style_params[name], pred_params[name])
                if not match:
                    mismatch_count += 1
                print(f"{name}: {'MATCH' if match else 'MISMATCH'}")
        
        if mismatch_count > 0:
            print(f"WARNING: {mismatch_count} parameter mismatches found!")

def check_data_loader():
    """Check if data loading introduces NaN"""
    from meldataset import build_dataloader
    
    print("\n=== Checking Data Loader ===")
    
    # Load config
    config = yaml.safe_load(open('Configs/config_multispeaker.yml'))
    device = config.get('device', 'cuda')
    
    # Build dataloader
    train_list_file = config.get('data_params', {}).get('train_data', '')
    
    if not train_list_file:
        print("No train data file specified")
        return
        
    train_dataloader = build_dataloader(
        train_list_file,
        batch_size=16,
        num_workers=1,
        device=device
    )
    
    # Check first batch
    for i, batch in enumerate(train_dataloader):
        if i >= 1:  # Only check first batch
            break
            
        print(f"\n=== Batch {i} ===")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                has_nan = torch.isnan(value).any().item()
                has_inf = torch.isinf(value).any().item()
                print(f"{key}: shape={value.shape}, dtype={value.dtype}, has_nan={has_nan}, has_inf={has_inf}")
                if value.numel() > 0:
                    print(f"  range=[{value.min():.3e}, {value.max():.3e}], mean={value.mean():.3e}")

if __name__ == "__main__":
    trace_nan_source()
    # check_data_loader()  # Uncomment if you want to check data loading