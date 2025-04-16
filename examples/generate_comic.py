import os
import argparse
from comic_generator import generate_comic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a comic book page")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--character_token", 
        type=str, 
        required=True,
        help="Token used for your character during training"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="comic_page.jpg", 
        help="Output filename for the comic page"
    )
    parser.add_argument(
        "--title", 
        type=str, 
        default="My Comic Adventure", 
        help="Title for the comic page"
    )
    
    args = parser.parse_args()
    
    # Define a sample story
    # In a real application, you might load this from a JSON file
    story = [
        {
            "prompt": f"a {args.character_token} waking up in a magical forest",
            "seed": 123,
            "description": "Our hero wakes up in a mysterious forest filled with glowing plants."
        },
        {
            "prompt": f"a {args.character_token} discovering a magical artifact",
            "seed": 456,
            "description": "They discover a strange crystal emitting a blue light."
        },
        {
            "prompt": f"a {args.character_token} confronted by a witch",
            "seed": 789,
            "description": "A mysterious witch appears, demanding the artifact."
        },
        {
            "prompt": f"a {args.character_token} running away with the magical artifact",
            "seed": 101,
            "description": "Our hero escapes with the artifact, determined to discover its secrets."
        }
    ]
    
    print(f"Generating comic page with title: {args.title}")
    
    # Generate the comic page
    comic_page = generate_comic(
        story=story,
        title=args.title,
        model_path=args.model_path
    )
    
    # Save the result
    comic_page.save(args.output)
    
    print(f"Comic page saved to: {args.output}")