"""
LoRA configuration for UI alignment detection models.
"""

from peft import LoraConfig

def get_ui_alignment_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
):
    """
    Get a LoRA configuration optimized for UI alignment detection.
    
    Args:
        r: Rank of the LoRA adapter
        lora_alpha: Scaling factor for the LoRA adapter
        lora_dropout: Dropout probability for the LoRA adapter
        bias: How to handle bias parameters ("none", "all", "lora_only")
        task_type: Task type for the adapter
        
    Returns:
        LoraConfig: Configuration for UI alignment LoRA training
    """
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention modules
            "gate_proj", "up_proj", "down_proj",     # MLP modules
            "lm_head"                                # Output head
        ]
    )