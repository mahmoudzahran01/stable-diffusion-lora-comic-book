import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
import textwrap
import os
from diffusers import StableDiffusionPipeline

from .lora import load_lora_models


def generate_images(
    prompt, 
    model_path,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base",
    num_samples=4, 
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=None,
    device="cuda"
):
    """
    Generate images using a fine-tuned Stable Diffusion model
    
    Args:
        prompt: Text prompt for image generation
        model_path: Path to the LoRA checkpoint
        pretrained_model_name_or_path: Base model path
        num_samples: Number of images to generate
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale for classifier-free guidance
        seed: Random seed for reproducibility
        device: Device to run generation on
        
    Returns:
        List of generated PIL Images
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Load base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    # Load LoRA weights
    pipe = load_lora_models(pipe, model_path)
    
    # Move to device and optimize if on GPU
    pipe.to(device)
    if device == "cuda":
        pipe.unet.half()
        pipe.text_encoder.half()
    
    # Generate images
    with torch.no_grad():
        images = pipe(
            [prompt] * num_samples, 
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale
        ).images
    
    return images


def create_comic_page(
    images, 
    descriptions, 
    title, 
    image_size=(300, 300), 
    font_path=None
):
    """
    Create a comic page layout from generated images
    
    Args:
        images: List of PIL Images
        descriptions: List of descriptions for each panel
        title: Title of the comic page
        image_size: Size to resize images to
        font_path: Path to font file (optional)
        
    Returns:
        PIL Image of the complete comic page
    """
    # Define layout parameters
    width, height = image_size
    border_size = 10  # Border around each image
    title_height = 100  # Title height
    gap_between_rows = 30  # Extra gap between rows
    desc_height = 50  # Height allocated for description under each image
    side_padding = 10  # White space reserved on both left and right sides
    column_padding = 20  # Padding between the first and second columns
    
    # Calculate total dimensions
    comic_width = 2 * (width + 2 * border_size) + column_padding + 2 * side_padding
    comic_height = 2 * height + title_height + gap_between_rows + 2 * desc_height + 4 * border_size
    
    # Create a new blank image for the comic page with white background
    comic_page = Image.new('RGB', (comic_width, comic_height), 'white')
    draw = ImageDraw.Draw(comic_page)
    
    # Font for the title and descriptions
    if font_path and os.path.exists(font_path):
        title_font = ImageFont.truetype(font_path, 60)  # Bigger font for title
        desc_font = ImageFont.truetype(font_path, 20)
    else:
        title_font = ImageFont.load_default()
        desc_font = ImageFont.load_default()
    
    # Draw the title at the top center of the comic page
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x_offset = (comic_width - title_width) // 2
    draw.text((title_x_offset, 20), title, fill='black', font=title_font)
    
    # Loop through images and descriptions
    for idx, (img, desc) in enumerate(zip(images, descriptions)):
        # Resize image to fit the grid and add border
        img = img.resize(image_size)
        img_with_border = ImageOps.expand(img, border=border_size, fill='black')
        
        # Calculate position in the grid (row and column)
        col = idx % 2
        row = idx // 2
        
        # Position of the image in the comic page
        x_offset = col * (width + 2 * border_size + column_padding // 2) + side_padding
        y_offset = row * (height + gap_between_rows + desc_height) + title_height
        
        # Paste the image with the border in the comic page
        comic_page.paste(img_with_border, (x_offset, y_offset))
        
        # Draw the description below each image
        desc_x_offset = x_offset + 10
        desc_y_offset = y_offset + height + border_size + 10
        
        desc_text = textwrap.fill(desc, width=47)  # Wrap text
        draw.text((desc_x_offset, desc_y_offset), desc_text, fill='black', font=desc_font)

    return comic_page


def generate_comic(
    story,
    title,
    model_path,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base",
    image_size=(300, 300),
    font_path=None,
    device="cuda"
):
    """
    Generate a complete comic page from a story description
    
    Args:
        story: List of dictionaries with 'prompt', 'seed', and 'description' keys
        title: Title of the comic
        model_path: Path to the LoRA checkpoint
        pretrained_model_name_or_path: Base model path
        image_size: Size of each panel
        font_path: Path to font file (optional)
        device: Device to run generation on
        
    Returns:
        PIL Image of the complete comic page
    """
    # Generate images for each panel
    images = []
    descriptions = []
    
    for panel in story:
        panel_images = generate_images(
            prompt=panel["prompt"],
            model_path=model_path,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            num_samples=4,  # Generate 4 options for each panel
            seed=panel.get("seed"),
            device=device
        )
        
        # Use the specified image index or default to the first one
        image_index = panel.get("image_index", 0)
        images.append(panel_images[image_index])
        descriptions.append(panel["description"])
    
    # Create the comic page
    return create_comic_page(
        images=images,
        descriptions=descriptions,
        title=title,
        image_size=image_size,
        font_path=font_path
    )