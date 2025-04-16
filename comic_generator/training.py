import os
import math
import itertools
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import pipeline

from .utils import load_models
from .dataset import DreamBoothDataset, collate_fn
from .lora import setup_lora


def train_model(
    image_dir,
    placeholder_token="<Character>",
    output_dir="./output",
    learning_rate=1e-4,
    max_train_steps=1000,
    train_batch_size=1,
    gradient_accumulation_steps=2,
    use_captioning=True,
    train_text_encoder=True,
    seed=42,
    device="cuda"
):
    """
    Train a Stable Diffusion model with LoRA on custom images
    
    Args:
        image_dir: Directory containing training images
        placeholder_token: Token to use for your character
        output_dir: Directory to save model
        learning_rate: Learning rate for training
        max_train_steps: Number of training steps
        train_batch_size: Batch size for training
        gradient_accumulation_steps: Number of steps to accumulate gradients
        use_captioning: Whether to use automatic captioning
        train_text_encoder: Whether to apply LoRA to text encoder
        seed: Random seed
        device: Device to train on
        
    Returns:
        Path to the trained model checkpoint
    """
    torch.manual_seed(seed)
    
    # Load the models
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_models(device=device)
    
    # Setup the dataset
    train_pipe = None
    if use_captioning:
        train_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=device)
    
    train_dataset = DreamBoothDataset(
        instance_data_root=image_dir,
        instance_prompt=f"a photo of {placeholder_token}",
        tokenizer=tokenizer,
        instance_prompt_model=train_pipe,
        instance_prompt_model_token=placeholder_token,
    )
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Apply LoRA
    unet, text_encoder = setup_lora(unet, text_encoder, train_text_encoder=train_text_encoder)
    
    # Setup training params
    hyperparameters = {
        "learning_rate": learning_rate,
        "max_train_steps": max_train_steps,
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "seed": seed,
        "output_dir": os.path.join(output_dir, f"__{placeholder_token}"),
        "train_text_encoder": train_text_encoder,
    }
    
    # Execute training
    _train(text_encoder, vae, unet, hyperparameters, train_dataset, noise_scheduler, device)
    
    # Return path to the final checkpoint
    final_checkpoint = os.path.join(
        hyperparameters["output_dir"], 
        f"checkpoint-{max_train_steps}_{learning_rate}"
    )
    
    return final_checkpoint


def _train(text_encoder, vae, unet, hyperparameters, train_dataset, noise_scheduler, device="cuda"):
    """
    Internal training function
    """
    torch.manual_seed(hyperparameters["seed"])

    train_batch_size = hyperparameters["train_batch_size"]
    gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
    learning_rate = hyperparameters["learning_rate"]
    max_train_steps = hyperparameters["max_train_steps"]
    output_dir = hyperparameters["output_dir"]
    gradient_checkpointing = hyperparameters["gradient_checkpointing"]
    train_text_encoder = hyperparameters["train_text_encoder"]

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        itertools.chain(unet.parameters(), text_encoder.parameters()) 
        if train_text_encoder else unet.parameters(),
        lr=learning_rate,
    )

    # Keep vae in eval mode as we don't train it
    vae.eval()
    vae.requires_grad_(False)
    
    # Move models to device
    vae.to(device)
    unet.to(device)
    text_encoder.to(device)

    # Set training mode
    unet.train()
    if train_text_encoder:
        text_encoder.train()
    else:
        text_encoder.eval()

    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Progress bar
    progress_bar = tqdm(range(max_train_steps))
    progress_bar.set_description("Steps")
    global_step = 0

    accumulated_loss = 0
    loss_running = []
    
    # Training loop
    for epoch in range(num_train_epochs):
        for i, batch in enumerate(train_dataloader):
            # Convert images to latent space
            batch["pixel_values"] = batch["pixel_values"].to(device, dtype=vae.dtype)
            latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device
            ).long()

            # Add noise to the latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Encode text embeddings
            batch["input_ids"] = batch["input_ids"].to(device)
            encoder_hidden_states = text_encoder(batch["input_ids"]).last_hidden_state

            # Predict the noise residual
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Calculate loss
            loss = torch.nn.functional.mse_loss(noise_pred, target)

            # Accumulate loss
            accumulated_loss = loss + accumulated_loss
            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.zero_grad()
                accumulated_loss.backward()

                optimizer.step()
                accumulated_loss = 0

                progress_bar.update(1)
                global_step += 1

                if global_step >= max_train_steps:
                    break

            # Log loss
            loss_running.append(loss.detach().item())
            logs = {"loss": sum(loss_running[-100:]) / min(len(loss_running), 100)}
            progress_bar.set_postfix(**logs)

    # Save the final model
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, 
        f"checkpoint-{global_step}_{hyperparameters['learning_rate']}"
    )
    os.makedirs(output_path, exist_ok=True)
    
    # Save UNet
    unet_save_path = os.path.join(output_path, "unet")
    os.makedirs(unet_save_path, exist_ok=True)
    unet.save_pretrained(unet_save_path, state_dict=unet.state_dict())
    
    # Save text encoder if trained
    if train_text_encoder:
        text_encoder_save_path = os.path.join(output_path, "text_encoder")
        os.makedirs(text_encoder_save_path, exist_ok=True)
        text_encoder.save_pretrained(text_encoder_save_path, state_dict=text_encoder.state_dict())