# Character Images Directory

Place your character reference images in this directory. The model will be trained to generate this character consistently across different scenes.

## Guidelines for Best Results

1. **Use 5-10 images**: Having a variety of poses and expressions helps the model learn the character better.

2. **Consistent style**: Try to use images with a consistent art style.

3. **Clean backgrounds**: Images with simple or removed backgrounds work best.

4. **Diversity**: Include different angles and poses to help the model generalize.

5. **Resolution**: Images should be at least 512×512 pixels for best results.

## Supported Formats

- JPG/JPEG
- PNG

## Example

After placing your images here, you can train your model with:

```bash
python examples/train_model.py --image_dir ./images --token "<YourCharacter>"
```