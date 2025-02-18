import os
import torch
from src.models import Decoder
from src.dataloader import get_flickr_dataloader
import argparse
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter("ignore", category=FutureWarning)


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    # Convert relative path to absolute path from project root
    if not os.path.isabs(checkpoint_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_path = os.path.join(project_root, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Use known training architecture
    model = Decoder(n_head=2, n_inner=512).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def generate_caption(model, image_embedding, tokenizer, max_length=77):
    """Generate a caption for an image."""
    model.eval()
    with torch.no_grad():
        # Start with empty sequence
        input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(image_embedding.device)
        # Generate tokens one by one
        for _ in range(max_length - 1):
            # Get next token probabilities
            log_probs = model(image_embedding, input_ids)
            next_token_logits = log_probs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)

            # Stop if we predict the end token
            if next_token.item() == tokenizer.eos_token_id:
                break

            # Add predicted token to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        # Decode the generated sequence
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
    args = parser.parse_args()

    # Load model from checkpoint
    model = load_model(args.checkpoint, device)

    # Get a single example from validation set with images and captions
    val_dataloader = get_flickr_dataloader(
        device, 
        split="val", 
        batch_size=1, 
        return_extras=True
    )
    batch = next(iter(val_dataloader))

    # Get a single example and generate caption
    image = batch["image"]
    original_caption = batch["caption"][0]  # Get first caption
    image_embedding = batch["image_embedding"].to(device)
    generated_caption = generate_caption(model, image_embedding, val_dataloader.dataset.tokenizer)

    print("generated_caption", generated_caption)
    print("original_caption", original_caption)

    # Display image and captions
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Generated: {generated_caption}\nGround truth: {original_caption}")
    plt.show()
