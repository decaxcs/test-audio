#!/usr/bin/env python3
"""
Data augmentation for StyleTTS2 to improve validation loss
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import random
from tqdm import tqdm

def augment_audio(audio, sr=24000, augmentation_type='all'):
    """Apply various augmentation techniques"""
    augmented = []
    
    # 1. Speed perturbation (0.9x to 1.1x)
    if augmentation_type in ['all', 'speed']:
        for speed_factor in [0.9, 0.95, 1.05, 1.1]:
            stretched = librosa.effects.time_stretch(audio, rate=speed_factor)
            augmented.append(('speed', speed_factor, stretched))
    
    # 2. Pitch shifting (-2 to +2 semitones)
    if augmentation_type in ['all', 'pitch']:
        for n_steps in [-2, -1, 1, 2]:
            pitched = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            augmented.append(('pitch', n_steps, pitched))
    
    # 3. Add slight noise
    if augmentation_type in ['all', 'noise']:
        for noise_factor in [0.001, 0.002]:
            noise = np.random.normal(0, noise_factor, len(audio))
            noisy = audio + noise
            noisy = np.clip(noisy, -1, 1)
            augmented.append(('noise', noise_factor, noisy))
    
    # 4. Volume perturbation
    if augmentation_type in ['all', 'volume']:
        for vol_factor in [0.8, 0.9, 1.1, 1.2]:
            volume_adjusted = audio * vol_factor
            volume_adjusted = np.clip(volume_adjusted, -1, 1)
            augmented.append(('volume', vol_factor, volume_adjusted))
    
    return augmented

def augment_dataset():
    """Augment the entire dataset"""
    data_dir = Path("Data/wavs")
    train_list_path = Path("Data/train_list.txt")
    
    # Read original training list
    with open(train_list_path, 'r') as f:
        original_lines = f.readlines()
    
    print(f"Original dataset size: {len(original_lines)} samples")
    
    augmented_lines = []
    augmented_count = 0
    
    # Process each original file
    for line in tqdm(original_lines, desc="Augmenting"):
        parts = line.strip().split('|')
        if len(parts) != 3:
            continue
            
        wav_path, text, speaker_id = parts
        full_wav_path = data_dir / wav_path
        
        if not full_wav_path.exists():
            print(f"Warning: {full_wav_path} not found")
            continue
        
        # Load audio
        audio, sr = librosa.load(full_wav_path, sr=24000)
        
        # Apply augmentations
        augmentations = augment_audio(audio, sr, 'all')
        
        # Save augmented versions
        for aug_type, param, aug_audio in augmentations:
            # Create augmented filename
            base_name = wav_path.replace('.wav', '')
            aug_filename = f"{base_name}_aug_{aug_type}_{str(param).replace('.', '_')}.wav"
            aug_path = data_dir / aug_filename
            
            # Save augmented audio
            sf.write(aug_path, aug_audio, sr)
            
            # Add to augmented list
            augmented_lines.append(f"{aug_filename}|{text}|{speaker_id}")
            augmented_count += 1
    
    # Create augmented training list
    augmented_train_path = Path("Data/train_list_augmented.txt")
    with open(augmented_train_path, 'w') as f:
        # Write original lines first
        f.writelines(original_lines)
        # Then augmented lines
        f.writelines([line + '\n' for line in augmented_lines])
    
    print(f"\nAugmentation complete!")
    print(f"Original samples: {len(original_lines)}")
    print(f"Augmented samples: {augmented_count}")
    print(f"Total samples: {len(original_lines) + augmented_count}")
    print(f"\nAugmented training list saved to: {augmented_train_path}")

def create_validation_augmented():
    """Create augmented validation set for better evaluation"""
    val_list_path = Path("Data/val_list.txt")
    
    with open(val_list_path, 'r') as f:
        val_lines = f.readlines()
    
    # For validation, only use mild augmentations
    augmented_val_lines = []
    data_dir = Path("Data/wavs")
    
    for line in val_lines:
        parts = line.strip().split('|')
        if len(parts) != 3:
            continue
            
        wav_path, text, speaker_id = parts
        full_wav_path = data_dir / wav_path
        
        if full_wav_path.exists():
            audio, sr = librosa.load(full_wav_path, sr=24000)
            
            # Only speed augmentation for validation
            for speed_factor in [0.95, 1.05]:
                stretched = librosa.effects.time_stretch(audio, rate=speed_factor)
                base_name = wav_path.replace('.wav', '')
                aug_filename = f"{base_name}_val_speed_{str(speed_factor).replace('.', '_')}.wav"
                aug_path = data_dir / aug_filename
                sf.write(aug_path, stretched, sr)
                augmented_val_lines.append(f"{aug_filename}|{text}|{speaker_id}")
    
    # Save augmented validation list
    augmented_val_path = Path("Data/val_list_augmented.txt")
    with open(augmented_val_path, 'w') as f:
        f.writelines(val_lines)
        f.writelines([line + '\n' for line in augmented_val_lines])
    
    print(f"Augmented validation list saved to: {augmented_val_path}")

if __name__ == "__main__":
    print("StyleTTS2 Data Augmentation Tool")
    print("=" * 40)
    
    # Check if augmentation already done
    if Path("Data/train_list_augmented.txt").exists():
        print("Augmented training list already exists!")
        response = input("Regenerate augmentations? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            exit()
    
    print("\n1. Augmenting training data...")
    augment_dataset()
    
    print("\n2. Creating augmented validation set...")
    create_validation_augmented()
    
    print("\nDone! Update your config to use:")
    print("  train_data: 'Data/train_list_augmented.txt'")
    print("  val_data: 'Data/val_list_augmented.txt'")