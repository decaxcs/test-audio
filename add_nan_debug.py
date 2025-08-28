#!/usr/bin/env python3
"""
Add comprehensive NaN debugging to train_second.py
"""

import re

# Read the train_second.py file
with open('train_second.py', 'r') as f:
    content = f.read()

# Add debugging function after imports
debug_code = '''
# ============ NaN DEBUGGING CODE START ============
def check_nan(tensor, name):
    """Check if tensor contains NaN and print details"""
    if tensor is None:
        return False
    if not isinstance(tensor, torch.Tensor):
        return False
    
    has_nan = torch.isnan(tensor).any().item()
    if has_nan:
        print(f"\\n!!! NaN detected in {name} !!!")
        print(f"Shape: {tensor.shape}")
        nan_mask = torch.isnan(tensor)
        print(f"NaN count: {nan_mask.sum().item()}/{tensor.numel()}")
        if tensor.dim() > 0:
            print(f"First NaN location: {torch.where(nan_mask)[0][0] if nan_mask.any() else 'None'}")
    return has_nan

def debug_model_state(model, prefix=""):
    """Check all model parameters for NaN"""
    nan_found = False
    for name, module in model.items():
        if hasattr(module, 'parameters'):
            for pname, param in module.named_parameters():
                if check_nan(param, f"{prefix}{name}.{pname}"):
                    nan_found = True
    return nan_found
# ============ NaN DEBUGGING CODE END ============

'''

# Insert after imports
import_end = content.find('@click.command()')
content = content[:import_end] + debug_code + content[import_end:]

# Add debugging after style encoder computation (around line 305-315)
style_debug = '''
                check_nan(mel, f"mel[{bib}]")
                check_nan(s, f"predictor_encoder_output[{bib}]")
'''

# Find where predictor encoder is called
pred_enc_pattern = r's = model\.predictor_encoder\(mel\.unsqueeze\(0\)\.unsqueeze\(1\)\)'
content = re.sub(pred_enc_pattern, 
                pred_enc_pattern + '\n' + style_debug, 
                content)

# Add debugging after decoder call
decoder_debug = '''
            
            # DEBUG: Check decoder inputs and outputs
            check_nan(en, "decoder_input_en")
            check_nan(F0_fake, "decoder_input_F0")
            check_nan(N_fake, "decoder_input_N")
            check_nan(s, "decoder_input_style")
            check_nan(y_rec, "decoder_output")
'''

# Find decoder call
decoder_pattern = r'y_rec = model\.decoder\(en, F0_fake, N_fake, s\)'
content = re.sub(decoder_pattern, 
                decoder_pattern + decoder_debug, 
                content)

# Add check at the beginning of training loop
training_loop_debug = '''
            # DEBUG: Check batch data
            if i == 0:  # First batch of epoch
                print(f"\\n=== Epoch {epoch}, Batch {i} ===")
                check_nan(mels, "input_mels")
                check_nan(texts, "input_texts")
                debug_model_state(model, "model.")
'''

# Find training loop start
loop_pattern = r'for i, batch in enumerate\(train_dataloader\):'
loop_match = re.search(loop_pattern, content)
if loop_match:
    insert_pos = loop_match.end()
    # Find next newline
    next_newline = content.find('\\n', insert_pos)
    content = content[:next_newline] + training_loop_debug + content[next_newline:]

# Add debugging for loss computation
loss_debug = '''
            
            # DEBUG: Check all loss components
            check_nan(loss_mel, "loss_mel")
            check_nan(loss_F0_rec, "loss_F0_rec")
            check_nan(loss_norm_rec, "loss_norm_rec")
            check_nan(loss_ce, "loss_ce")
            check_nan(loss_dur, "loss_dur")
            if 'loss_gen_all' in locals():
                check_nan(loss_gen_all, "loss_gen_all")
            check_nan(g_loss, "g_loss_total")
'''

# Find g_loss computation
g_loss_pattern = r'g_loss = loss_params\.lambda_mel \* loss_mel'
g_loss_match = re.search(g_loss_pattern, content)
if g_loss_match:
    # Find the end of the g_loss computation (next line)
    insert_pos = content.find('\\n', g_loss_match.end())
    content = content[:insert_pos] + loss_debug + content[insert_pos:]

# Write the modified file
with open('train_second_debug.py', 'w') as f:
    f.write(content)

print("Created train_second_debug.py with comprehensive NaN debugging")
print("\\nThe debug version will:")
print("1. Check all input batches for NaN")
print("2. Check model parameters at the start of each epoch")  
print("3. Check intermediate computations (encoder outputs, decoder inputs/outputs)")
print("4. Check all loss components before backpropagation")
print("\\nRun with:")
print("cd /root/StyleTTS2")
print("python train_second_debug.py --config_path ./Configs/config_multispeaker.yml 2>&1 | tee debug_log.txt")