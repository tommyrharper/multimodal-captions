import os
import torch
from src.models import Transformer
from src.dataloader import get_flickr_dataloader
import argparse
import warnings
import matplotlib.pyplot as plt
import textwrap

warnings.simplefilter("ignore", category=FutureWarning)


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    # Convert relative path to absolute path from project root
    if not os.path.isabs(checkpoint_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_path = os.path.join(project_root, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Use known training architecture
    model = Transformer(n_head=2, n_inner=512).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def generate_caption(model, image_embedding, tokenizer, min_length=5):
    """Generate a caption for an image."""
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(image_embedding.device)

        for i in range(77 - 1):  # Fixed max length of 77 from training
            log_probs = model(image_embedding, input_ids)
            next_token_logits = log_probs[:, -1, :]

            # Force non-EOS tokens for first min_length tokens
            if i < min_length:
                next_token_logits[0, tokenizer.eos_token_id] = float("-inf")

            next_token = torch.argmax(next_token_logits, dim=-1)

            # Stop if we predict the end token (after min_length)
            if next_token.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        caption = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return caption


if __name__ == "__main__":
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device", device)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model_epoch_9.pt",
        help="Path to checkpoint relative to project root or absolute path",
    )
    parser.add_argument(
        "--image-id",
        type=int,
        default=0,
        help="Index of validation image to caption (default: 0)",
    )
    args = parser.parse_args()

    # Load model from checkpoint
    model = load_model(args.checkpoint, device)

    # Get validation dataloader
    val_dataloader = get_flickr_dataloader(
        device, split="val", batch_size=1, return_extras=True
    )

    # Get the requested image
    for i, batch in enumerate(val_dataloader):
        if i == args.image_id:
            break
    else:
        print(f"Error: Image index {args.image_id} is out of range")
        exit(1)

    # Generate caption
    image = batch["image"]
    original_caption = batch["caption"][0]
    image_embedding = batch["image_embedding"].to(device)
    generated_caption = generate_caption(
        model, image_embedding, val_dataloader.dataset.tokenizer
    )

    print("Generated caption:", generated_caption)
    print("Original caption:", original_caption)

    # Wrap captions for display
    wrapped_gen = textwrap.fill(f"Generated: {generated_caption}", width=60)
    wrapped_orig = textwrap.fill(f"Ground truth: {original_caption}", width=60)

    # Display image and wrapped captions
    plt.figure(figsize=(10, 10))  # Make figure taller
    plt.imshow(image)
    plt.axis("off")

    # Add title with smaller font and more padding
    plt.title(
        f"{wrapped_gen}\n\n{wrapped_orig}",
        pad=20,
        fontsize=10,
        wrap=True,
        y=1.05,  # Move title up
    )

    # Adjust layout to prevent text cutoff
    plt.subplots_adjust(top=0.85)  # Leave more space at top
    plt.show()
