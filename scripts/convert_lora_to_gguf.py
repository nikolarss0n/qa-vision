#!/usr/bin/env python3
"""
Script to convert a PEFT/LoRA adapter trained with HuggingFace
to a format compatible with llama-cpp-python for deployment.
"""

import os
import json
import struct
import numpy as np
import torch
from pathlib import Path
import argparse
import sys

# Ensure this script can be run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default paths
DEFAULT_ADAPTER_PATH = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         "output", "ui_alignment_lora_adapter"))

def parse_args():
    """Parse command line arguments for LoRA conversion."""
    parser = argparse.ArgumentParser(description="Convert PEFT/LoRA adapters to GGML/GGUF for llama.cpp")
    
    parser.add_argument("--input", type=str, default=str(DEFAULT_ADAPTER_PATH),
                        help="Directory containing the adapter")
    parser.add_argument("--output", type=str, help="Output GGUF file path")
    parser.add_argument("--base-model", type=str, help="Base model name (for configuration)")
    parser.add_argument("--scale", type=float, default=1.0, 
                        help="Scaling factor for the adapter")
    
    return parser.parse_args()


def load_adapter(adapter_path):
    """
    Load a PEFT/LoRA adapter from disk.
    
    Args:
        adapter_path: Path to the adapter directory
        
    Returns:
        tuple: Adapter weights and configuration
    """
    adapter_path = Path(adapter_path)
    
    # Check if adapter_model.bin exists
    bin_path = adapter_path / "adapter_model.bin"
    if not bin_path.exists():
        print(f"Error: {bin_path} does not exist")
        return None, None
    
    # Load adapter config
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        print(f"Error: {config_path} does not exist")
        return None, None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load adapter weights
    adapter_weights = torch.load(bin_path, map_location="cpu")
    
    return adapter_weights, config


def convert_to_gguf(adapter_weights, config, output_path, scale=1.0):
    """
    Convert adapter weights to GGUF format.
    
    Args:
        adapter_weights: Adapter weights
        config: Adapter configuration
        output_path: Output file path
        scale: Scaling factor for the adapter
        
    Returns:
        bool: Success status
    """
    try:
        # This is a simplified placeholder for the actual conversion
        # In a real implementation, this would involve more complex GGUF format handling
        # For now, we'll just save the adapter in a custom format with metadata
        
        # Extract LoRA configuration
        lora_alpha = config.get("lora_alpha", 1.0)
        lora_r = config.get("r", 16)
        target_modules = config.get("target_modules", [])
        
        # Process weights
        processed_weights = {}
        for key, value in adapter_weights.items():
            # Skip non-tensor values
            if not isinstance(value, torch.Tensor):
                continue
            
            # Convert to numpy
            processed_weights[key] = {
                "data": value.cpu().numpy(),
                "shape": value.shape,
                "dtype": str(value.dtype)
            }
        
        # Create metadata
        metadata = {
            "format_version": 1,
            "lora_alpha": lora_alpha,
            "lora_r": lora_r,
            "scale": scale,
            "target_modules": target_modules,
            "weight_keys": list(processed_weights.keys())
        }
        
        # Save in a custom format (simplified)
        output_data = {
            "metadata": metadata,
            "weights": processed_weights
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save as a pytorch file for simplicity
        torch.save(output_data, output_path)
        
        print(f"Adapter converted and saved to {output_path}")
        print(f"Note: This is a simplified conversion, not a true GGUF file.")
        print(f"For production use with llama.cpp, additional conversion steps would be needed.")
        
        return True
    
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False


def main():
    """Main function to convert LoRA adapters."""
    args = parse_args()
    
    # Load adapter
    print(f"Loading adapter from {args.input}")
    adapter_weights, config = load_adapter(args.input)
    
    if adapter_weights is None or config is None:
        print("Failed to load adapter. Exiting.")
        return
    
    # Determine output path
    if not args.output:
        output_path = f"{args.input}/adapter_model.gguf"
    else:
        output_path = args.output
    
    # Convert adapter
    success = convert_to_gguf(
        adapter_weights=adapter_weights,
        config=config,
        output_path=output_path,
        scale=args.scale
    )
    
    if success:
        print("Conversion completed successfully")
    else:
        print("Conversion failed")


if __name__ == "__main__":
    main()