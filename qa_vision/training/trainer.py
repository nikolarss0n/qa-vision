"""
Training utilities for UI alignment detection.
"""

import os
import torch
from datetime import datetime
from pathlib import Path
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer
)
from peft import get_peft_model

from ..models.model_loader import load_model_for_training
from ..data.dataset import prepare_dataset, MultimodalDataCollator
from .lora_config import get_ui_alignment_lora_config

def setup_training_args(
    output_dir: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    epochs: int,
    warmup_steps: int,
    use_cuda: bool,
    fp16: bool,
    bf16: bool,
    gradient_checkpointing: bool,
    model_name: str
):
    """
    Set up training arguments for the Trainer.
    
    Args:
        output_dir: Directory to save model checkpoints
        batch_size: Batch size for training
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        epochs: Number of epochs
        warmup_steps: Number of warmup steps
        use_cuda: Whether CUDA is available and enabled
        fp16: Whether to use FP16 precision
        bf16: Whether to use BF16 precision
        gradient_checkpointing: Whether to enable gradient checkpointing
        model_name: Name of the model (for run name)
        
    Returns:
        TrainingArguments: Training arguments for the Trainer
    """
    # Create a descriptive run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_short = model_name.split("/")[-1]
    run_name = f"{model_name_short}-lora-ui-alignment-{timestamp}"
    
    # Configure mixed precision training
    use_fp16 = use_cuda and fp16
    use_bf16 = use_cuda and bf16 and torch.cuda.is_bf16_supported()
    
    # High-performance training configuration
    return TrainingArguments(
        output_dir=f"{output_dir}/{run_name}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,  # Can use larger eval batch size
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        warmup_steps=warmup_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=use_fp16 and not use_bf16,  # Use fp16 if enabled and bf16 not enabled
        bf16=use_bf16,                   # Use bf16 if supported and enabled
        report_to="none",
        run_name=run_name,
        logging_steps=50,
        gradient_checkpointing=gradient_checkpointing,
        # Optimizer settings for better convergence
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        # Save disk space by removing checkpoints as we go
        save_total_limit=2,
    )


def train_ui_alignment_lora(
    model_name: str,
    output_dir: str,
    train_data_path: str,
    test_data_path: str,
    batch_size: int = 4,
    epochs: int = 5,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    gradient_accumulation_steps: int = 1,
    max_length: int = 512,
    cpu_only: bool = False,
    subset_size: int = 0,
    fp16: bool = True,
    bf16: bool = False,
    gradient_checkpointing: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05
):
    """
    Train a LoRA adapter for UI alignment detection.
    
    Args:
        model_name: HuggingFace model name or path to local model
        output_dir: Directory to save model checkpoints
        train_data_path: Path to training data
        test_data_path: Path to test data
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        gradient_accumulation_steps: Gradient accumulation steps
        max_length: Maximum sequence length
        cpu_only: Whether to force CPU training
        subset_size: Size of the subset to use (0 for full dataset)
        fp16: Whether to use FP16 precision
        bf16: Whether to use BF16 precision
        gradient_checkpointing: Whether to enable gradient checkpointing
        lora_r: Rank of the LoRA adapter
        lora_alpha: Scaling factor for the LoRA adapter
        lora_dropout: Dropout probability for the LoRA adapter
        
    Returns:
        Tuple: Trained model and processor
    """
    # Setup CUDA availability
    use_cuda = torch.cuda.is_available() and not cpu_only
    
    # Load model and processor
    print(f"Loading model: {model_name}")
    model, processor = load_model_for_training(
        model_name=model_name,
        use_cuda=use_cuda,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing
    )
    
    # Get LoRA configuration
    lora_config = get_ui_alignment_lora_config(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )
    
    # Apply LoRA to the model
    print("Applying LoRA adapter")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Determine if we're using a multimodal model
    is_multimodal = "llava" in model_name.lower()
    
    # Prepare datasets
    print("Preparing datasets")
    train_dataset = prepare_dataset(
        data_path=train_data_path,
        processor=processor,
        max_length=max_length,
        is_multimodal=is_multimodal,
        subset_size=subset_size
    )
    test_dataset = prepare_dataset(
        data_path=test_data_path,
        processor=processor,
        max_length=max_length,
        is_multimodal=is_multimodal,
        subset_size=subset_size
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Setup training arguments
    training_args = setup_training_args(
        output_dir=output_dir,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        epochs=epochs,
        warmup_steps=warmup_steps,
        use_cuda=use_cuda,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        model_name=model_name
    )
    
    # Initialize data collator
    data_collator = MultimodalDataCollator()
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor if isinstance(processor, AutoTokenizer) else processor.tokenizer,
        data_collator=data_collator,
    )
    
    # Train the model
    print("Starting training")
    trainer.train()
    
    # Save the final model
    run_name = training_args.run_name
    final_output_dir = f"{output_dir}/{run_name}/final"
    trainer.save_model(final_output_dir)
    
    print(f"Training complete. Model saved to {final_output_dir}")
    
    # Save LoRA adapter separately for easier reuse
    lora_output_dir = f"{output_dir}/ui_alignment_lora_adapter"
    model.save_pretrained(lora_output_dir)
    print(f"LoRA adapter saved to {lora_output_dir}")
    
    return model, processor