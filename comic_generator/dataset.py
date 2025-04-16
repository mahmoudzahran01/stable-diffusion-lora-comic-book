import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from transformers import Pipeline
from .utils import tokenize_prompt


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the images with the prompts for fine-tuning the model.
    It pre-processes the images and tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        tokenizer,
        size=512,
        instance_prompt=None,
        instance_prompt_model=None,
        instance_prompt_model_token=None,
    ):
        self.size = size
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        
        # Add this line to filter for only image files
        self.instance_images_path = sorted([p for p in Path(instance_data_root).iterdir() 
                                        if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
        
        self.num_instance_images = len(self.instance_images_path)
        if self.num_instance_images == 0:
            raise ValueError(f"No image files found in {instance_data_root}")
            
        if instance_prompt_model is not None:
            self.instance_prompt = self.obtain_caption_for_each_image(
                instance_prompt_model, self.instance_images_path, instance_prompt_model_token)
        else:
            assert instance_prompt is not None
            self.instance_prompt = instance_prompt
            
        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def obtain_caption_for_each_image(self, pipeline, image_paths, token):
        """Generate captions for each image using the provided pipeline"""
        return [f'an image of {token}.' + ' ' + pipeline(Image.open(path))[0]['generated_text'] 
                for path in image_paths] 

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
            
        example["instance_images"] = self.image_transforms(instance_image)
        
        if isinstance(self.instance_prompt, str):
            instance_prompt = self.instance_prompt
        elif isinstance(self.instance_prompt, list) and isinstance(self.instance_prompt[0], str):
            instance_prompt = self.instance_prompt[index]
        else:
            raise ValueError("Instance prompt must be a string or a list of strings")

        example["instance_prompt_ids"] = tokenize_prompt(self.tokenizer, instance_prompt)
        example["instance_prompt"] = instance_prompt

        return example


def collate_fn(examples):
    """
    Collate function to combine multiple examples into a batch
    """
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch