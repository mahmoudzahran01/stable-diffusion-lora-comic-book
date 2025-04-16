from setuptools import setup, find_packages

setup(
    name="comic_generator",
    version="0.1.0",
    description="Generate comic book pages using Stable Diffusion with LoRA",
    author="Mahmoud Zahran",
    author_email="your.email@example.com",  # Replace with your email
    url="https://github.com/mahmoudzahran01/stable-diffusion-lora-comic-book",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "diffusers>=0.14.0",
        "transformers>=4.25.1",
        "Pillow>=9.3.0",
        "tqdm>=4.64.1",
        "peft>=0.2.0",
        "accelerate>=0.16.0",
        "safetensors>=0.3.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)