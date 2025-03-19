#!/usr/bin/env python3
"""
Command-line script for preparing UI alignment datasets for training.
"""

import os
import json
import random
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ensure this script can be run directly
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define default paths
DEFAULT_OUTPUT_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

# Caption templates for different types of alignment issues
CAPTION_TEMPLATES = {
    "aligned": [
        "This is a well-aligned UI element with proper spacing and alignment.",
        "The UI component shown has correct alignment and follows design best practices.",
        "This screenshot displays a properly aligned {element_type} with good visual hierarchy.",
        "This is an example of a correctly aligned {element_type} with appropriate spacing.",
        "The {element_type} in this image demonstrates proper alignment principles."
    ],
    "misaligned": [
        "This UI element has alignment issues: {description}",
        "The {element_type} shown has poor alignment with {issue_type} problems.",
        "This screenshot displays a misaligned {element_type} that needs adjustment.",
        "This UI component has alignment issues, specifically {issue_type}.",
        "This screenshot shows a {element_type} with poor alignment. The issue is: {description}"
    ]
}

# Instruction templates for model training
INSTRUCTION_TEMPLATES = [
    "Analyze this UI screenshot and identify any alignment issues.",
    "Is this UI properly aligned? If not, explain the alignment issues.",
    "What alignment problems can you identify in this UI screenshot?",
    "Evaluate the visual alignment of elements in this UI screenshot.",
    "Check this UI for alignment issues and explain any problems you find."
]

def parse_args():
    """Parse command line arguments for dataset preparation."""
    parser = argparse.ArgumentParser(description="Prepare UI alignment datasets for training")
    
    parser.add_argument("--dataset_dir", type=str, required=True,
                       help="Directory containing the UI alignment dataset")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                       help="Directory to save the prepared datasets")
    parser.add_argument("--annotations_file", type=str, default="annotations.json",
                       help="Filename of the annotations file (within dataset directory)")
    parser.add_argument("--test_size", type=float, default=0.15,
                       help="Fraction of data to use for testing")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    return parser.parse_args()


def prepare_captions(row, is_aligned):
    """
    Prepare captions for a UI screenshot.
    
    Args:
        row: Row from the annotations dataframe
        is_aligned: Whether the UI is properly aligned
        
    Returns:
        str: Generated caption
    """
    templates = CAPTION_TEMPLATES["aligned" if is_aligned else "misaligned"]
    template = random.choice(templates)
    
    # Default values
    element_type = row.get('element_type', 'UI component')
    issue_type = row.get('issue_type', 'alignment')
    description = row.get('description', 'elements are not properly aligned')
    
    # Format template with available information
    return template.format(
        element_type=element_type,
        issue_type=issue_type,
        description=description
    )


def create_instruction_example(row, caption):
    """
    Create an instruction-tuning example.
    
    Args:
        row: Row from the annotations dataframe
        caption: Caption for the UI screenshot
        
    Returns:
        dict: Instruction-tuning example
    """
    instruction = random.choice(INSTRUCTION_TEMPLATES)
    
    return {
        "image": row["image_path"],
        "text": f"<s>[INST] {instruction} [/INST] {caption}</s>",
        "metadata": {
            "is_aligned": row.get("is_aligned", False),
            "element_type": row.get("element_type", ""),
            "issue_type": row.get("issue_type", "")
        }
    }


def main():
    """Main function to prepare datasets."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define paths
    dataset_dir = Path(args.dataset_dir)
    annotations_file = dataset_dir / args.annotations_file
    
    print(f"Loading annotations from {annotations_file}")
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data['annotations'])
    
    # Add full path to images
    df['image_path'] = df['file_path'].apply(lambda x: str(dataset_dir / x))
    
    # Generate captions
    df['caption'] = df.apply(
        lambda row: prepare_captions(row, row.get('is_aligned', False)),
        axis=1
    )
    
    # Create instruction examples
    examples = []
    for _, row in df.iterrows():
        examples.append(create_instruction_example(row, row['caption']))
    
    # Split into train and test sets
    train_examples, test_examples = train_test_split(
        examples, 
        test_size=args.test_size, 
        random_state=args.seed
    )
    
    print(f"Created {len(train_examples)} training examples and {len(test_examples)} test examples")
    
    # Save training data
    train_file = Path(args.output_dir) / "train.jsonl"
    with open(train_file, 'w') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
    
    # Save test data
    test_file = Path(args.output_dir) / "test.jsonl"
    with open(test_file, 'w') as f:
        for example in test_examples:
            f.write(json.dumps(example) + '\n')
    
    # Save instruction templates
    instructions_file = Path(args.output_dir) / "train_instructions.json"
    with open(instructions_file, 'w') as f:
        json.dump({
            "instructions": INSTRUCTION_TEMPLATES,
            "captions": CAPTION_TEMPLATES
        }, f, indent=2)
    
    print(f"Datasets saved to {args.output_dir}")
    print(f"  - Training data: {train_file}")
    print(f"  - Test data: {test_file}")
    print(f"  - Instructions: {instructions_file}")


if __name__ == "__main__":
    main()