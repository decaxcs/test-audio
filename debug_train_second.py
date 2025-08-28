#!/usr/bin/env python3
"""
Enhanced train_second.py with detailed NaN debugging
"""

import torch
import numpy as np

def check_tensor(tensor, name, return_stats=False):
    """Check tensor for NaN/Inf and extreme values"""
    if tensor is None:
        return f"{name}: None"
    
    if not isinstance(tensor, torch.Tensor):
        return f"{name}: Not a tensor (type: {type(tensor)})"
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if tensor.numel() > 0:
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        mean_val = tensor.mean().item()
        std_val = tensor.std().item() if tensor.numel() > 1 else 0
    else:
        min_val = max_val = mean_val = std_val = 0
    
    status = []
    if has_nan:
        status.append("HAS NaN")
    if has_inf:
        status.append("HAS Inf")
    if abs(max_val) > 1e10:
        status.append("EXTREME VALUES")
    
    result = f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, "
    result += f"min={min_val:.3e}, max={max_val:.3e}, mean={mean_val:.3e}, std={std_val:.3e}"
    
    if status:
        result += f" [{'|'.join(status)}]"
    
    if return_stats:
        return result, has_nan, has_inf, max_val
    return result

def add_nan_debugging(train_second_path):
    """Add NaN debugging to train_second.py"""
    
    # Read the original file
    with open(train_second_path, 'r') as f:
        content = f.read()
    
    # Insert debugging code after imports
    debug_functions = '''
# === START NaN DEBUGGING FUNCTIONS ===
import torch
import numpy as np

def check_tensor(tensor, name, print_always=False):
    """Check tensor for NaN/Inf and extreme values"""
    if tensor is None:
        if print_always:
            print(f"DEBUG: {name} is None")
        return False
    
    if not isinstance(tensor, torch.Tensor):
        if print_always:
            print(f"DEBUG: {name} is not a tensor (type: {type(tensor)})")
        return False
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if tensor.numel() > 0:
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        mean_val = tensor.mean().item()
        abs_max = tensor.abs().max().item()
    else:
        min_val = max_val = mean_val = abs_max = 0
    
    # Always print if NaN/Inf or extreme values
    should_print = has_nan or has_inf or abs_max > 1e10 or print_always
    
    if should_print:
        status = []
        if has_nan:
            status.append("NaN")
        if has_inf:
            status.append("Inf")
        if abs_max > 1e10:
            status.append("Extreme")
        
        print(f"DEBUG {name}: shape={tensor.shape}, range=[{min_val:.3e}, {max_val:.3e}], mean={mean_val:.3e}, max_abs={abs_max:.3e} {status}")
    
    return has_nan or has_inf

def debug_checkpoint_loading():
    """Debug checkpoint after loading"""
    print("\\n=== DEBUGGING CHECKPOINT LOADING ===")
    global model
    
    for key in ['style_encoder', 'predictor_encoder', 'decoder', 'text_encoder']:
        if key in model:
            print(f"\\nChecking {key}:")
            if hasattr(model[key], 'parameters'):
                param_count = 0
                nan_params = 0
                for name, param in model[key].named_parameters():
                    param_count += 1
                    if torch.isnan(param).any():
                        nan_params += 1
                        print(f"  NaN in {name}: {param.shape}")
                print(f"  Total parameters: {param_count}, with NaN: {nan_params}")

# === END NaN DEBUGGING FUNCTIONS ===

'''
    
    # Find where to insert (after imports)
    import_end = content.find('\nif __name__ == "__main__":')
    if import_end == -1:
        import_end = content.find('\n\ndef build_model')
    
    # Insert debugging functions
    content = content[:import_end] + '\n' + debug_functions + content[import_end:]
    
    # Add debugging after checkpoint loading
    checkpoint_load_pattern = 'print("Copied trained style_encoder weights to predictor_encoder")'
    checkpoint_load_pos = content.find(checkpoint_load_pattern)
    if checkpoint_load_pos != -1:
        insert_pos = checkpoint_load_pos + len(checkpoint_load_pattern)
        content = content[:insert_pos] + '\n            debug_checkpoint_loading()\n' + content[insert_pos:]
    
    # Add debugging in training loop - find where loss is computed
    # Look for the main loss computation section
    loss_computation_pattern = 'y_rec_gt_pred = decoder'
    loss_pos = content.find(loss_computation_pattern)
    
    if loss_pos != -1:
        # Find the end of that line
        line_end = content.find('\n', loss_pos)
        debug_code = '''
            
            # === DEBUG: Check all tensors before loss computation ===
            if i < 3:  # Only debug first few batches
                print(f"\\n=== DEBUG Batch {i} ===")
                check_tensor(mels, "mels")
                check_tensor(texts, "texts") 
                check_tensor(input_lengths, "input_lengths")
                check_tensor(mel_input_length, "mel_input_length")
                check_tensor(pitches, "pitches")
                check_tensor(pitch_masks, "pitch_masks")
                
                # Check style encoder output
                check_tensor(s_trg, "s_trg (style target)")
                check_tensor(s, "s (style vector)")
                
                # Check predictor outputs  
                check_tensor(en, "en (encoder output)")
                check_tensor(F0_pred, "F0_pred")
                check_tensor(N_pred, "N_pred")
                check_tensor(dur, "dur")
                
                # Check decoder input/output
                check_tensor(asr, "asr (alignment)")
                check_tensor(decoder_inputs, "decoder_inputs", print_always=True)
                check_tensor(y_rec_gt_pred, "y_rec_gt_pred (decoder output)", print_always=True)
'''
        content = content[:line_end] + debug_code + content[line_end:]
    
    # Add check before loss backward
    loss_all_pattern = 'loss_all = loss_mel'
    loss_all_pos = content.find(loss_all_pattern)
    if loss_all_pos != -1:
        line_end = content.find('\n', loss_all_pos)
        debug_loss = '''
            
            # === DEBUG: Check loss components ===
            if i < 3:
                print(f"\\n=== Loss Components Batch {i} ===")
                check_tensor(loss_mel, "loss_mel")
                check_tensor(loss_s2s, "loss_s2s") 
                check_tensor(loss_mono, "loss_mono")
                check_tensor(loss_gen_all, "loss_gen_all") if 'loss_gen_all' in locals() else None
                check_tensor(loss_d, "loss_d") if 'loss_d' in locals() else None
'''
        content = content[:line_end] + debug_loss + content[line_end:]
    
    return content

# Create the debug version
print("Creating debug version of train_second.py...")
debug_content = add_nan_debugging('/mnt/c/Users/Deca/Desktop/Testing/StyleTTS2/train_second.py')

# Write to new file
with open('/mnt/c/Users/Deca/Desktop/Testing/StyleTTS2/train_second_debug.py', 'w') as f:
    f.write(debug_content)

print("Created train_second_debug.py with enhanced debugging")
print("\nRun with:")
print("cd /root/StyleTTS2") 
print("python train_second_debug.py --config_path ./Configs/config_multispeaker.yml")