# Directory Structure

```
stable-diffusion-lora-comic-book/
├── .gitignore
├── README.md
├── DIRECTORY_STRUCTURE.md
├── requirements.txt
├── setup.py
├── comic_generator/
│   ├── __init__.py
│   ├── utils.py
│   ├── dataset.py
│   ├── lora.py
│   ├── training.py
│   └── generation.py
├── examples/
│   ├── train_model.py
│   └── generate_comic.py
└── images/
    └── README.md  # Placeholder for image directory
```

## Directory Descriptions

- `comic_generator/`: Main package containing all the core functionality
- `examples/`: Example scripts demonstrating how to use the library
- `images/`: Directory to place your character images for training

## Usage

1. Place your character images in the `images/` directory
2. Run the training script: `python examples/train_model.py --image_dir ./images --token "<YourCharacter>"`
3. Generate a comic page: `python examples/generate_comic.py --model_path ./output/__<YourCharacter>/checkpoint-1000_0.0001 --character_token "<YourCharacter>"`