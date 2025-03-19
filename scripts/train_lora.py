#!/usr/bin/env python3
"""
Command-line script for training a UI alignment LoRA adapter.
"""

import os
import argparse
from pathlib import Path

# Ensure this script can be run directly
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa_vision.utils.device import setup_device_optimizations
from qa_vision.training.trainer import train_ui_alignment_lora

# Define paths
DEFAULT_DATA_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
DEFAULT_OUTPUT_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))

def parse_args():
    """Parse command line arguments for LoRA training."""
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for UI alignment detection")
    
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf", 
                        help="HuggingFace model name or path to local model")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    
    parser.add_argument("--train_data", type=str, 
                        default=str(DEFAULT_DATA_DIR / "train.jsonl"),
                        help="Path to training data")
    parser.add_argument("--test_data", type=str, 
                        default=str(DEFAULT_DATA_DIR / "test.jsonl"),
                        help="Path to test data")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory to save model checkpoints")
    
    # Hardware options
    parser.add_argument("--cpu_only", action="store_true",
                        help="Force CPU training")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Enable mixed precision training")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Enable bfloat16 mixed precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="Enable gradient checkpointing to save memory")
    
    # Dataset options
    parser.add_argument("--subset_size", type=int, default=0,
                        help="Use subset of data for testing, 0 for all data")
    
    # LoRA options
    parser.add_argument("--lora_r", type=int, default=16,
                        help="Rank of the LoRA adapter")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="Scaling factor for the LoRA adapter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout probability for the LoRA adapter")
    
    return parser.parse_args()


def main():
    """Main function to run LoRA training."""
    # Setup device optimizations
    setup_device_optimizations()
    
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data directory if it doesn't exist
    train_data_dir = os.path.dirname(args.train_data)
    test_data_dir = os.path.dirname(args.test_data)
    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Train the model
    train_ui_alignment_lora(
        model_name=args.model_name,
        output_dir=args.output_dir,
        train_data_path=args.train_data,
        test_data_path=args.test_data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        cpu_only=args.cpu_only,
        subset_size=args.subset_size,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )


if __name__ == "__main__":
    main()