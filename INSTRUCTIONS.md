# Getting Started with QA Vision

I've combined the UI alignment tools into a comprehensive package called "QA Vision" with improved code organization, flexibility, and additional features.

## Quick Start

1. First, navigate to the qa-vision directory and initialize the git repository:

```bash
cd /Users/nklars0/ProjectsAI/qa-vision
chmod +x *.sh scripts/*.py
./init_repo.sh
```

2. Set up the project (creates virtual environment and installs dependencies):

```bash
./setup_project.sh
```

3. Run the all-in-one script for help:

```bash
./run_qa_vision.sh help
```

## Training a UI Alignment Model

To train a LoRA adapter for UI alignment detection:

```bash
./run_qa_vision.sh train \
  --model_name llava-hf/llava-1.5-7b-hf \
  --batch_size 4 \
  --epochs 5 \
  --train_data /path/to/train.jsonl \
  --test_data /path/to/test.jsonl
```

For high-end GPU training (NVIDIA RTX 4080 Super):

```bash
./run_qa_vision.sh train \
  --model_name llava-hf/llava-1.5-7b-hf \
  --batch_size 4 \
  --epochs 5 \
  --bf16
```

## Analyzing UI Screenshots

To analyze UI screenshots for alignment issues:

```bash
./run_qa_vision.sh analyze --image /path/to/screenshot.png
```

Or analyze an entire directory:

```bash
./run_qa_vision.sh analyze --dir /path/to/screenshots --visualize
```

## Project Structure

The project is organized as follows:

- `qa_vision/`: Core package with reusable modules
  - `models/`: Model loading and inference utilities
  - `training/`: Training utilities and configurations
  - `data/`: Dataset processing and preparation
  - `utils/`: Utility functions and helpers
- `scripts/`: Command-line scripts
  - `train_lora.py`: Training script for LoRA adapters
  - `analyze_ui.py`: Analyze UI screenshots
  - `inference.py`: Run inference on individual images
  - `prepare_dataset.py`: Prepare datasets for training
  - `convert_lora_to_gguf.py`: Convert LoRA to GGUF format
- `data/`: Directory for datasets
- `output/`: Directory for trained models and adapters

## Next Steps

1. Consider uploading your trained adapters to HuggingFace Hub for easy sharing
2. Customize the UI alignment detection prompts to match your specific needs
3. Integrate the package into your CI/CD pipeline for automated UI testing