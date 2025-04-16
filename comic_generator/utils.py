import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel, pipeline
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel

def image_grid(imgs, rows, cols, resize=256):
    """
    Create a grid of images for display
    
    Args:
        imgs: List of PIL images
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        resize: Size to resize images to (default: 256)
        
    Returns:
        PIL Image containing the grid
    """
    assert len(imgs) == rows * cols

    if resize is not None:
        imgs = [img.resize((resize, resize)) for img in imgs]
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def tokenize_prompt(tokenizer, prompt):
    """
    Tokenize a text prompt
    
    Args:
        tokenizer: HuggingFace tokenizer
        prompt: Text prompt to tokenize
        
    Returns:
        Tensor of input IDs
    """
    return tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True).input_ids


def decode_tokens(tokenizer, tokens):
    """
    Decode token IDs back to text
    
    Args:
        tokenizer: HuggingFace tokenizer
        tokens: Tensor of token IDs
        
    Returns:
        Decoded text string
    """
    return tokenizer.decode(tokens, skip_special_tokens=True)


def caption_image(image, device="cuda"):
    """
    Generate a caption for an image using BLIP captioning model
    
    Args:
        image: PIL Image to caption
        device: Device to run the model on
        
    Returns:
        Caption string
    """
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=device)
    return pipe(image)[0]["generated_text"]


def load_models(pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base", device="cuda"):
    """
    Load all necessary models for Stable Diffusion
    
    Args:
        pretrained_model_name_or_path: Path to pretrained model
        device: Device to load models on
        
    Returns:
        Tuple of (tokenizer, text_encoder, vae, unet, noise_scheduler)
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
        device=device
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="text_encoder"
    ).to(device)
    
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="vae"
    ).to(device)
    
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="unet"
    ).to(device)

    # Using DDIM scheduler for more efficient sampling
    noise_scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    return tokenizer, text_encoder, vae, unet, noise_scheduler