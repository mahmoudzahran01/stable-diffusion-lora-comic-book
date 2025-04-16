# Comic Book Generation with Stable Diffusion and LoRA

This repository contains code for generating comic book pages using fine-tuned Stable Diffusion models with Low-Rank Adaptation (LoRA) techniques.

## Overview

Diffusion models, especially in combination with text-to-image conditioning, are excellent for creating diverse environments and backgrounds for artistic creation. This project uses techniques like DreamBooth and LoRA to generate high-quality, consistent comic book pages with minimal input.

- **DreamBooth** ([paper](https://arxiv.org/abs/2208.12242)) for Character Consistency: Allows you to add specific characters to a pre-trained model by fine-tuning with just a few images, ensuring consistency across different scenes.
- **Low-Rank Adaptation (LoRA)** ([paper](https://arxiv.org/abs/2106.09685)) for Style and Atmosphere: Fine-tune models to adapt specific stylistic elements without retraining the entire model, maintaining cohesive visual experiences.

## Features

- Fine-tune Stable Diffusion models with character images
- Generate consistent characters across different comic panels
- Create complete comic book pages with custom storylines
- Automatic image captioning for enriched prompts

## Installation

```bash
# Clone the repository
git clone https://github.com/mahmoudzahran01/stable-diffusion-lora-comic-book
cd stable-diffusion-lora-comic-book

# Install in development mode
pip install -e .
```

## Usage

### Training a model

```python
from comic_generator import train_model

# Train a model with your character images
train_model(
    image_dir="./images",
    placeholder_token="<YourCharacter>",
    learning_rate=1e-4,
    max_train_steps=1000
)
```

### Generating a comic page

```python
from comic_generator import generate_comic

# Define your story
story = [
    {
        "prompt": "a glowing crystal in a dark forest, there's a <YourCharacter> next to it",
        "seed": 777,
        "description": "The character stumbles upon a glowing crystal in the forest."
    },
    # Add more panels...
]

# Generate the comic page
comic_page = generate_comic(
    story=story,
    title="The Adventure Begins",
    model_path="./output/your_model_checkpoint"
)

# Save the comic page
comic_page.save("my_comic.jpg")
```

## Example Results

![UnicornGirl Comic Book](https://github.com/mahmoudzahran01/stable-diffusion-lora-comic-book/raw/main/examples/unicorn_adventure.jpg)

This comic page was generated using a character model trained on just a few images. The story follows a UnicornGirl character as she:
1. Wakes up in a mysterious forest filled with glowing plants
2. Discovers a strange crystal emitting a blue light
3. Confronts a mysterious witch demanding the artifact
4. Escapes with the artifact, determined to discover its secrets

## Using the Command Line

You can also use the provided command line scripts:

```bash
# Train a model
python examples/train_model.py --image_dir ./images --token "<UnicornGirl>" --steps 1000

# Generate a comic
python examples/generate_comic.py --model_path ./output/__\<UnicornGirl\>/checkpoint-1000_0.0001 --character_token "<UnicornGirl>" --title "The UnicornGirl Adventure" --output unicorn_adventure.jpg
```

## License

MIT