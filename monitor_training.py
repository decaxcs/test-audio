#!/usr/bin/env python3
"""
Comprehensive training monitor for StyleTTS2
Shows progress, estimates completion time, and alerts on issues
"""

import re
import time
import os
from datetime import datetime, timedelta
import subprocess
import sys

class TrainingMonitor:
    def __init__(self, log_path="Models/ThreeSpeaker/train.log"):
        self.log_path = log_path
        self.start_time = None
        self.validation_losses = []
        self.epochs = []
        self.best_loss = float('inf')
        self.best_epoch = 0
        
    def parse_validation_loss(self, line):
        """Extract validation loss from log line"""
        # Pattern: "Validation loss: 0.572"
        match = re.search(r'Validation loss: ([\d.]+)', line)
        if match:
            return float(match.group(1))
        return None
        
    def parse_epoch(self, line):
        """Extract epoch number from log line"""
        # Pattern: "Epochs: 150"
        match = re.search(r'Epochs: (\d+)', line)
        if match:
            return int(match.group(1))
        return None
    
    def get_gpu_usage(self):
        """Get current GPU memory usage"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                used, total, util = result.stdout.strip().split(', ')
                return float(used), float(total), float(util)
        except:
            return None, None, None
        
    def estimate_completion(self, current_epoch, target_epoch, elapsed_time):
        """Estimate training completion time"""
        if current_epoch == 0:
            return "Calculating..."
        
        time_per_epoch = elapsed_time / current_epoch
        remaining_epochs = target_epoch - current_epoch
        remaining_time = time_per_epoch * remaining_epochs
        
        completion_time = datetime.now() + timedelta(seconds=remaining_time)
        return completion_time.strftime("%Y-%m-%d %H:%M:%S")
    
    def check_training_health(self):
        """Check for training issues"""
        issues = []
        
        if len(self.validation_losses) > 5:
            # Check if loss is increasing
            recent_losses = self.validation_losses[-5:]
            if all(recent_losses[i] > recent_losses[i-1] for i in range(1, len(recent_losses))):
                issues.append("⚠️  Validation loss increasing for 5 epochs!")
            
            # Check if loss is stagnant
            if len(set(recent_losses)) == 1:
                issues.append("⚠️  Validation loss not changing!")
            
            # Check if loss is too high after many epochs
            if self.epochs[-1] > 100 and self.validation_losses[-1] > 0.8:
                issues.append("⚠️  Loss still high after 100+ epochs!")
                
        return issues
    
    def monitor(self):
        """Main monitoring loop"""
        print("StyleTTS2 Training Monitor")
        print("=" * 60)
        print(f"Log file: {self.log_path}")
        print("Press Ctrl+C to stop monitoring\n")
        
        if not os.path.exists(self.log_path):
            print(f"Waiting for {self.log_path} to be created...")
            while not os.path.exists(self.log_path):
                time.sleep(5)
        
        # Open log file
        with open(self.log_path, 'r') as f:
            # Skip to end of file
            f.seek(0, 2)
            
            self.start_time = time.time()
            last_update = 0
            
            while True:
                line = f.readline()
                
                if line:
                    # Parse epoch
                    epoch = self.parse_epoch(line)
                    if epoch is not None:
                        self.epochs.append(epoch)
                    
                    # Parse validation loss
                    val_loss = self.parse_validation_loss(line)
                    if val_loss is not None:
                        self.validation_losses.append(val_loss)
                        
                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            self.best_epoch = self.epochs[-1] if self.epochs else 0
                
                # Update display every second
                if time.time() - last_update > 1:
                    self.display_status()
                    last_update = time.time()
                
                time.sleep(0.1)
    
    def display_status(self):
        """Display current training status"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("StyleTTS2 Training Monitor")
        print("=" * 60)
        
        # Training time
        if self.start_time:
            elapsed = time.time() - self.start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
        else:
            elapsed_str = "00:00:00"
            elapsed = 0
        
        print(f"Training Time: {elapsed_str}")
        
        # Current status
        if self.epochs:
            current_epoch = self.epochs[-1]
            print(f"Current Epoch: {current_epoch}/400")
            
            # Progress bar
            progress = current_epoch / 400
            bar_length = 40
            filled = int(bar_length * progress)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"Progress: [{bar}] {progress*100:.1f}%")
            
            # ETA
            eta = self.estimate_completion(current_epoch, 400, elapsed)
            print(f"Estimated Completion: {eta}")
        else:
            print("Current Epoch: Waiting for data...")
        
        print("\n" + "-" * 60)
        
        # Validation loss
        if self.validation_losses:
            current_loss = self.validation_losses[-1]
            print(f"Current Validation Loss: {current_loss:.4f}")
            print(f"Best Validation Loss: {self.best_loss:.4f} (Epoch {self.best_epoch})")
            
            # Trend
            if len(self.validation_losses) > 1:
                trend = self.validation_losses[-1] - self.validation_losses[-2]
                trend_symbol = "↓" if trend < 0 else "↑" if trend > 0 else "→"
                print(f"Trend: {trend_symbol} {abs(trend):.4f}")
        else:
            print("Validation Loss: Waiting for first validation...")
        
        print("\n" + "-" * 60)
        
        # GPU status
        mem_used, mem_total, gpu_util = self.get_gpu_usage()
        if mem_used is not None:
            mem_percent = (mem_used / mem_total) * 100
            print(f"GPU Memory: {mem_used:.0f}/{mem_total:.0f} MB ({mem_percent:.1f}%)")
            print(f"GPU Utilization: {gpu_util:.0f}%")
            
            if mem_percent > 95:
                print("⚠️  GPU memory usage critical!")
        
        print("\n" + "-" * 60)
        
        # Health checks
        issues = self.check_training_health()
        if issues:
            print("Issues Detected:")
            for issue in issues:
                print(issue)
        else:
            print("Training Status: ✅ Healthy")
        
        # Recent losses
        if len(self.validation_losses) > 0:
            print("\nRecent Validation Losses:")
            recent = list(zip(self.epochs[-5:], self.validation_losses[-5:]))
            for epoch, loss in recent:
                print(f"  Epoch {epoch:3d}: {loss:.4f}")
        
        print("\n" + "=" * 60)
        print("Press Ctrl+C to stop monitoring")

def main():
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = "Models/ThreeSpeaker/train.log"
    
    monitor = TrainingMonitor(log_path)
    
    try:
        monitor.monitor()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print(f"Best validation loss: {monitor.best_loss:.4f} at epoch {monitor.best_epoch}")

if __name__ == "__main__":
    main()