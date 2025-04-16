import os
import argparse
from comic_generator import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom comic character model")
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=True, 
        help="Directory containing character images"
    )
    parser.add_argument(
        "--token", 
        type=str, 
        default="<Character>", 
        help="Token to use for your character"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./output", 
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-4, 
        help="Learning rate for training"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=1000, 
        help="Number of training steps"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="Training batch size"
    )
    parser.add_argument(
        "--no_captioning", 
        action="store_true", 
        help="Disable automatic image captioning"
    )
    parser.add_argument(
        "--no_text_encoder", 
        action="store_true", 
        help="Disable training of text encoder"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    print(f"Training model for character token: {args.token}")
    print(f"Using images from: {args.image_dir}")
    
    checkpoint_path = train_model(
        image_dir=args.image_dir,
        placeholder_token=args.token,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_train_steps=args.steps,
        train_batch_size=args.batch_size,
        use_captioning=not args.no_captioning,
        train_text_encoder=not args.no_text_encoder,
        seed=args.seed
    )
    
    print(f"Training complete! Model saved to: {checkpoint_path}")