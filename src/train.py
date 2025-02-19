import torch
from torch import nn
from src.dataloader import get_flickr_dataloader
from src.models import Transformer
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
    batch_size=32,
    num_layers=6,
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
            "batch_size": batch_size,
            "num_layers": num_layers,
        }
    )

    train_dataloader = get_flickr_dataloader(
        device, split="train", batch_size=batch_size
    )
    val_dataloader = get_flickr_dataloader(device, split="val", batch_size=batch_size)

    transformer = Transformer(n_head=num_heads, n_inner=num_inner, num_layers=num_layers).to(device)
    optimizer = optim.AdamW(transformer.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
        transformer.train()
        total_train_loss = 0
        num_train_batches = 0
        train_iter = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")

        for batch_idx, batch in enumerate(train_iter, 1):
            loss = compute_loss(batch, transformer, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches = batch_idx
            postfix = logger.log_batch(
                loss.item(), total_train_loss, batch_idx, epoch, "train"
            )
            train_iter.set_postfix(postfix)

            if max_batches != 0 and batch_idx >= max_batches:
                break

        # Validation
        transformer.eval()
        total_val_loss = 0
        num_val_batches = 0
        val_iter = tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}")

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_iter, 1):
                loss = compute_loss(batch, transformer, device)
                total_val_loss += loss.item()
                num_val_batches = batch_idx
                postfix = logger.log_batch(
                    loss.item(), total_val_loss, batch_idx, epoch, "val"
                )
                val_iter.set_postfix(postfix)

                if max_batches != 0 and batch_idx >= max_batches:
                    break

        # Calculate correct averages using respective batch counts
        average_train_loss = total_train_loss / num_train_batches
        average_val_loss = total_val_loss / num_val_batches
        logger.log_epoch(average_train_loss, average_val_loss, epoch)

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            logger.save_checkpoint(
                transformer, 
                epoch, 
                average_train_loss, 
                average_val_loss,
                config={
                    'n_head': num_heads,
                    'n_inner': num_inner,
                    'num_layers': num_layers,
                }
            )
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
    parser.add_argument(
        "--lr",
        type=int,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=2,
        help="Number of heads to use (default: 2)",
    )
    parser.add_argument(
        "--num-inner",
        type=int,
        default=512,
        help="Number of inner dimensions (default: 512)",
    )
    parser.add_argument(
        "--weight-decay",
        type=int,
        default=0.01,
        help="Weight decay level (default: 0.01)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of decoder layers (default: 6)",
    )
    args = parser.parse_args()

    train(
        device=device,
        num_heads=args.num_heads,
        num_inner=args.num_inner,
        use_wandb=args.wandb,
        num_epochs=args.epochs,
        max_batches=args.max_batches,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
    )
