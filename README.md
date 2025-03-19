# QA Vision

QA Vision is a tool that helps detect and analyze UI alignment issues in screenshots. It combines computer vision and large language models to provide detailed feedback on interface design problems.

## Features

- **UI Alignment Detection**: Identify misaligned UI elements, spacing issues, and layout problems
- **Visual Analysis**: Process screenshots to detect visual inconsistencies
- **LLM-based Feedback**: Generate detailed reports with suggested fixes
- **LoRA Adapter Training**: Fine-tune models specifically for UI/UX quality assurance

## Setup

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/qa-vision.git
cd qa-vision

# Install dependencies
pip install -r requirements.txt

# Optional: Install additional dependencies for model training
pip install -r requirements-training.txt
```

## Usage

### UI Alignment Analysis

```bash
python analyze_ui.py --image_path "/path/to/screenshot.png"
```

### Training a UI Alignment LoRA

```bash
python train_lora.py --model_name llava-hf/llava-1.5-7b-hf --batch_size 4 --epochs 5
```

### Inference with Trained Model

```bash
python inference.py --image_path "/path/to/screenshot.png"
```

## Model Training

The project includes specialized LoRA adapters for UI alignment detection, trained on datasets of well-aligned and misaligned UI examples.

See [TRAINING.md](TRAINING.md) for detailed instructions on training models for different environments.

## License

[MIT License](LICENSE)