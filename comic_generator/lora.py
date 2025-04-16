from peft import LoraConfig, get_peft_model, PeftModel


def setup_lora(unet, text_encoder, train_text_encoder=False, verbose=False):
    """
    Apply LoRA adaptations to the UNet and optionally the text encoder
    
    Args:
        unet: UNet model from Stable Diffusion
        text_encoder: Text encoder model
        train_text_encoder: Whether to apply LoRA to the text encoder
        verbose: Whether to print model architectures
        
    Returns:
        Tuple of (unet, text_encoder) with LoRA adaptations applied
    """
    # Define LoRA parameters
    lora_rank_unet = 4  # Low-rank dimension for UNet
    lora_alpha_unet = 32  # Scaling factor for UNet
    lora_rank_text_encoder = 8  # Low-rank dimension for text encoder
    lora_alpha_text_encoder = 64  # Scaling factor for text encoder

    # Configure LoRA for UNet
    config = LoraConfig(
        r=lora_rank_unet,
        lora_alpha=lora_alpha_unet,
        target_modules=["to_k", "to_q", "to_v", "proj_out"],
    )

    if verbose:
        print("Original UNet parameters before LoRA:")
        for name, m in unet.named_modules():
            print(f"module {name}: {m}")
        
        print("\nOriginal Text Encoder parameters before LoRA:")
        for name, m in text_encoder.named_modules():
            print(f"module {name}: {m}")

    # Apply LoRA to UNet
    unet = get_peft_model(unet, config)
    
    if verbose:
        unet.print_trainable_parameters()

    # Apply LoRA to text encoder if requested
    if train_text_encoder:
        config = LoraConfig(
            r=lora_rank_text_encoder,
            lora_alpha=lora_alpha_text_encoder,
            target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],
        )

        text_encoder = get_peft_model(text_encoder, config)
        
        if verbose:
            text_encoder.print_trainable_parameters()

    if verbose:
        print("\nUNet parameters after applying LoRA:")
        for name, m in unet.named_modules():
            print(f"module {name}: {m}")
        
        print("\nText Encoder parameters after applying LoRA:")
        for name, m in text_encoder.named_modules():
            print(f"module {name}: {m}")

    return unet, text_encoder


def load_lora_models(base_pipeline, checkpoint_path):
    """
    Load trained LoRA weights into a pipeline
    
    Args:
        base_pipeline: StableDiffusionPipeline to load weights into
        checkpoint_path: Path to the checkpoint with LoRA weights
        
    Returns:
        Updated pipeline with LoRA weights
    """
    unet_sub_dir = f"{checkpoint_path}/unet"
    text_encoder_sub_dir = f"{checkpoint_path}/text_encoder"

    # Load LoRA weights into the pipeline's models
    base_pipeline.unet = PeftModel.from_pretrained(
        base_pipeline.unet, 
        unet_sub_dir, 
        adapter_name="comic_lora"
    )
    
    base_pipeline.text_encoder = PeftModel.from_pretrained(
        base_pipeline.text_encoder, 
        text_encoder_sub_dir, 
        adapter_name="comic_lora"
    )
    
    return base_pipeline