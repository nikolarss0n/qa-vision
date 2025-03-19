# Training Models for QA Vision

This document provides detailed instructions for training specialized LoRA adapters for the QA Vision project.

## UI Alignment LoRA Training

### Hardware Requirements

- **GPU Training**: CUDA-compatible GPU with at least 8GB VRAM (16GB+ recommended)
- **CPU Training**: Possible but extremely slow (32GB+ RAM recommended)

### Environment Setup

1. Create a conda environment (recommended):
   ```bash
   conda create -n qa_vision python=3.10
   conda activate qa_vision
   ```

2. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. Install training dependencies:
   ```bash
   pip install -r requirements-training.txt
   ```

4. Login to Hugging Face (if using gated models):
   ```bash
   huggingface-cli login
   ```

### Data Preparation

The training process uses a dataset of UI screenshots labeled with alignment issues:

```bash
python prepare_dataset.py
```

This script:
1. Processes UI screenshots
2. Creates training and test splits
3. Formats data with appropriate annotations
4. Saves prepared datasets

### Training Parameters

The `train_lora.py` script accepts several parameters:

```
--model_name MODEL_NAME   HuggingFace model name or path to local model
--batch_size BATCH_SIZE   Batch size for training (default: 4)
--epochs EPOCHS           Number of training epochs (default: 5)
--learning_rate LR        Learning rate (default: 2e-5)
--gradient_accumulation_steps STEPS  Gradient accumulation steps (default: 1)
--max_length MAX_LENGTH   Maximum sequence length (default: 512)
--output_dir OUTPUT_DIR   Directory to save the trained adapter
--fp16                    Enable mixed precision training
--bf16                    Enable bfloat16 training (RTX 4000 series)
--gradient_checkpointing  Enable gradient checkpointing to save memory
--subset_size SIZE        Use subset of data for testing (0 for all data)
```

### Training Configurations

#### Basic Training (NVIDIA RTX 3060 12GB or similar)

```bash
python train_lora.py \
  --model_name llava-hf/llava-1.5-7b-hf \
  --batch_size 2 \
  --epochs 3 \
  --gradient_accumulation_steps 4 \
  --fp16
```

#### High-End GPU Training (NVIDIA RTX 4080/4090 or similar)

```bash
python train_lora.py \
  --model_name llava-hf/llava-1.5-7b-hf \
  --batch_size 4 \
  --epochs 5 \
  --bf16
```

#### Memory-Optimized Training (For Larger Models)

```bash
python train_lora.py \
  --model_name llava-hf/llava-1.5-13b-hf \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing \
  --fp16
```

### Training Process

The training process:

1. Loads a base model (LLaVA or similar)
2. Applies LoRA configuration
3. Processes the UI alignment dataset
4. Fine-tunes the model
5. Saves the trained adapter

### After Training

After training completes:
1. The full model will be saved at `output/{timestamp}/final`
2. The LoRA adapter will be saved at `output/ui_alignment_lora_adapter`

## Inference with Trained Models

To test a trained model:

```bash
python inference.py --image_path "/path/to/screenshot.png"
```

## Performance Monitoring

- For NVIDIA GPUs: Use `nvidia-smi -l 1` to monitor GPU usage
- For CPU usage: Use system monitors or tools like `htop`

## Troubleshooting

**Out of memory errors:**
- Reduce batch size
- Enable gradient checkpointing
- Reduce sequence length
- Use model quantization

**Poor training results:**
- Increase number of epochs
- Adjust learning rate
- Check dataset quality and diversity