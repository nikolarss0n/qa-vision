"""
Device and memory management utilities.
"""

import os
import gc
import torch

def setup_device_optimizations():
    """
    Set up device-specific optimizations and memory management.
    
    Returns:
        str: Name of the available device ('cuda', 'mps', or 'cpu')
    """
    # Memory management setup for different devices
    if torch.cuda.is_available():
        # CUDA setup for NVIDIA GPUs
        device = 'cuda'
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Enable TF32 precision for faster training on Ampere+ GPUs (RTX 3000+)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Pre-allocate memory to avoid fragmentation
        torch.cuda.empty_cache()
        
    elif torch.backends.mps.is_available():
        # Apple Silicon GPU (M1/M2/M3)
        device = 'mps'
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable upper limit
        print("MPS backend available, setting high watermark ratio to 0.0")
        
    else:
        # CPU fallback
        device = 'cpu'
        print("Neither CUDA nor MPS available, using CPU")

    # Force garbage collection to free memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return device