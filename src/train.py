import torch
from torch import nn
from src.dataloader import get_flickr_dataloader
from src.models import Decoder
from torch import optim
import random
import numpy as np
from tqdm import tqdm
from src.logger import TrainingLogger
import argparse


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_loss(batch, model, device):
    """Compute loss for a single batch."""
    image_embedding = batch["image_embedding"].to(device)
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    log_probs = model(image_embedding, input_ids)
    loss = nn.functional.nll_loss(
        log_probs.reshape(-1, model.config.vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )
    return loss


def train(
    device,
    num_heads=4,
    num_inner=1024,
    num_epochs=1,
    lr=1e-4,
    weight_decay=0.01,
    use_wandb=False,
    max_batches=0,
):
    set_seed()

    # Initialize logger
    logger = TrainingLogger(use_wandb=use_wandb)
    logger.log_config(
        {
            "num_heads": num_heads,
            "num_inner": num_inner,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "max_batches": max_batches,
        }
    )

    train_dataloader = get_flickr_dataloader(device, split="train")
    val_dataloader = get_flickr_dataloader(device, split="val")

    decoder = Decoder(n_head=num_heads, n_inner=num_inner).to(device)
    optimizer = optim.AdamW(decoder.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
        decoder.train()
        total_train_loss = 0
        train_iter = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")

        for batch_idx, batch in enumerate(train_iter, 1):
            loss = compute_loss(batch, decoder, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            postfix = logger.log_batch(
                loss.item(), total_train_loss, batch_idx, epoch, "train"
            )
            train_iter.set_postfix(postfix)

            if max_batches != 0 and batch_idx >= max_batches:
                break

        # Validation
        decoder.eval()
        total_val_loss = 0
        val_iter = tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}")

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_iter, 1):
                loss = compute_loss(batch, decoder, device)
                total_val_loss += loss.item()
                postfix = logger.log_batch(
                    loss.item(), total_val_loss, batch_idx, epoch, "val"
                )
                val_iter.set_postfix(postfix)

                if max_batches != 0 and batch_idx >= max_batches:
                    break

        # Log epoch results and save checkpoint if improved
        average_train_loss = total_train_loss / batch_idx
        average_val_loss = total_val_loss / batch_idx
        logger.log_epoch(average_train_loss, average_val_loss, epoch)

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            logger.save_checkpoint(decoder, epoch, average_train_loss, average_val_loss)
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")

    logger.finish()


if __name__ == "__main__":
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device", device)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Number of batches to train (default: all)",
    )
    args = parser.parse_args()

    train(
        device=device,
        num_heads=2,
        num_inner=512,
        use_wandb=args.wandb,
        num_epochs=args.epochs,
        max_batches=args.max_batches,
    )
