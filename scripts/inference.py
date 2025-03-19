#!/usr/bin/env python3
"""
Command-line script for running inference with trained UI alignment models.
"""

import os
import argparse
import torch
from pathlib import Path

# Ensure this script can be run directly
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa_vision.utils.device import setup_device_optimizations
from qa_vision.models.inference import load_model_for_inference, generate_ui_alignment_analysis

# Define default paths
DEFAULT_ADAPTER_PATH = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         "output", "ui_alignment_lora_adapter"))

def parse_args():
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(description="Run inference with a UI alignment model")
    
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf", 
                        help="HuggingFace model name or path to local model")
    parser.add_argument("--adapter_path", type=str, default=str(DEFAULT_ADAPTER_PATH),
                        help="Path to the LoRA adapter")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the UI screenshot image")
    parser.add_argument("--prompt", type=str, 
                        default="Analyze this UI screenshot and identify any alignment issues.",
                        help="Prompt to use for generation")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p for nucleus sampling")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k for sampling")
    
    # Hardware options
    parser.add_argument("--cpu_only", action="store_true",
                        help="Force CPU inference")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use half-precision (float16) for inference")
    
    return parser.parse_args()


def main():
    """Main function to run inference."""
    # Setup device optimizations
    device = setup_device_optimizations()
    
    # Parse arguments
    args = parse_args()
    
    # Determine device and precision
    if args.cpu_only:
        device = "cpu"
        dtype = torch.float32
    else:
        if device == "cuda" and args.fp16:
            dtype = torch.float16
        else:
            dtype = torch.float32
    
    # Check if adapter exists
    if not os.path.exists(args.adapter_path) and args.adapter_path == str(DEFAULT_ADAPTER_PATH):
        print(f"LoRA adapter not found at {args.adapter_path}. Using base model.")
        adapter_path = None
    else:
        adapter_path = args.adapter_path
    
    # Load model and processor
    model, processor = load_model_for_inference(
        model_name=args.model_name,
        adapter_path=adapter_path,
        device=device,
        torch_dtype=dtype
    )
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Image not found: {args.image_path}")
        return
    
    # Generate analysis
    print(f"Analyzing image: {args.image_path}\n")
    analysis = generate_ui_alignment_analysis(
        model=model,
        processor=processor,
        image_path=args.image_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
    
    # Print the analysis
    print("UI Alignment Analysis:")
    print("=" * 50)
    print(analysis)
    print("=" * 50)


if __name__ == "__main__":
    main()