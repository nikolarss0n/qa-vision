#!/usr/bin/env python3
"""
Command-line script for analyzing UI screenshots for alignment issues.
"""

import os
import argparse
import json
import glob
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Ensure this script can be run directly
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa_vision.utils.device import setup_device_optimizations
from qa_vision.models.inference import load_model_for_inference, generate_ui_alignment_analysis

# Define default paths
DEFAULT_ADAPTER_PATH = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         "output", "ui_alignment_lora_adapter"))

def parse_args():
    """Parse command line arguments for UI analysis."""
    parser = argparse.ArgumentParser(description="Analyze UI screenshots for alignment issues")
    
    # Input options
    parser.add_argument("--image", type=str, help="Path to a specific image to analyze")
    parser.add_argument("--dir", type=str, help="Directory of images to analyze")
    
    # Output options
    parser.add_argument("--output", type=str, default="ui_analysis_results.json",
                        help="Output file for results")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize results with matplotlib")
    
    # Model options
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf", 
                        help="HuggingFace model name or path to local model")
    parser.add_argument("--adapter_path", type=str, default=str(DEFAULT_ADAPTER_PATH),
                        help="Path to the LoRA adapter")
    
    # Prompt options
    parser.add_argument("--prompt", type=str, 
                        default="Analyze this UI screenshot and identify any alignment issues.",
                        help="Prompt to use for generation")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation")
    
    # Hardware options
    parser.add_argument("--cpu_only", action="store_true",
                        help="Force CPU inference")
    
    return parser.parse_args()


def analyze_image(model, processor, image_path, prompt, max_new_tokens, temperature):
    """
    Analyze a single image for UI alignment issues.
    
    Args:
        model: The model to use for analysis
        processor: The processor for the model
        image_path: Path to the image to analyze
        prompt: Prompt to use for generation
        max_new_tokens: Maximum tokens to generate
        temperature: Temperature for generation
        
    Returns:
        dict: Analysis results
    """
    print(f"Analyzing {image_path}...")
    
    # Generate the analysis
    analysis = generate_ui_alignment_analysis(
        model=model,
        processor=processor,
        image_path=image_path,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    
    # Extract severity score from the analysis if available
    severity_keywords = {
        "no issues": 0.0,
        "minor": 0.3,
        "moderate": 0.6,
        "severe": 0.9,
        "critical": 1.0
    }
    
    # Default severity score
    severity = 0.0
    
    # Check for severity keywords in the analysis
    for keyword, score in severity_keywords.items():
        if keyword in analysis.lower():
            severity = max(severity, score)
    
    # Determine if there are alignment issues
    has_issues = "no issues" not in analysis.lower() and "properly aligned" not in analysis.lower()
    if has_issues:
        severity = max(0.3, severity)  # At least minor if issues exist
    
    # Create result object
    result = {
        "image_path": str(image_path),
        "analysis": analysis,
        "has_alignment_issues": has_issues,
        "severity_score": severity
    }
    
    return result


def visualize_results(results):
    """
    Visualize analysis results with matplotlib.
    
    Args:
        results: List of analysis results
    """
    # Create a figure with subplots based on number of results
    n_images = len(results)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    # Handle single row or column case
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flat
    
    # Plot each image with its severity score
    for i, result in enumerate(results):
        if i < len(axes):
            # Load and display the image
            img = Image.open(result["image_path"])
            axes[i].imshow(img)
            
            # Set title with severity indicator
            title = f"Severity: {result['severity_score']:.2f}"
            axes[i].set_title(title, color='red' if result['severity_score'] > 0.5 else 'black')
            
            # Set axis off
            axes[i].axis('off')
    
    # Turn off remaining subplot axes
    for i in range(len(results), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to analyze UI screenshots."""
    # Setup device optimizations
    device = setup_device_optimizations()
    
    # Parse arguments
    args = parse_args()
    
    # Check required arguments
    if not args.image and not args.dir:
        print("Either --image or --dir must be specified")
        return
    
    # Check if adapter exists
    if not os.path.exists(args.adapter_path) and args.adapter_path == str(DEFAULT_ADAPTER_PATH):
        print(f"LoRA adapter not found at {args.adapter_path}. Using base model.")
        adapter_path = None
    else:
        adapter_path = args.adapter_path
    
    # Override device if CPU is forced
    if args.cpu_only:
        device = "cpu"
    
    # Load model and processor
    model, processor = load_model_for_inference(
        model_name=args.model_name,
        adapter_path=adapter_path,
        device=device
    )
    
    # Collect images to analyze
    images = []
    if args.image:
        if os.path.exists(args.image):
            images.append(Path(args.image))
        else:
            print(f"Image not found: {args.image}")
            return
    
    if args.dir:
        if os.path.exists(args.dir):
            # Find all images in directory
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                images.extend(Path(args.dir).glob(ext))
        else:
            print(f"Directory not found: {args.dir}")
            return
    
    # Analyze images
    results = []
    for image_path in images:
        result = analyze_image(
            model=model,
            processor=processor,
            image_path=image_path,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        results.append(result)
    
    # Print summary of results
    print("\nAnalysis Summary:")
    print("=" * 50)
    for result in results:
        status = "Has Issues" if result["has_alignment_issues"] else "No Issues"
        print(f"{os.path.basename(result['image_path'])}: {status} (Severity: {result['severity_score']:.2f})")
    
    # Save results to JSON
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    # Visualize results if requested
    if args.visualize and results:
        visualize_results(results)


if __name__ == "__main__":
    main()