import torch
from torch import nn
from src.dataloader import get_flickr_dataloader
from src.models import Decoder
from torch import optim
import random
import numpy as np
from tqdm import tqdm
import wandb

debug_batch_num = 3


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


def log_batch_metrics(wandb, loss, total_loss, batch_idx, epoch, split="train"):
    """Log metrics for a single batch."""
    running_avg_loss = total_loss / batch_idx
    wandb.log(
        {
            f"{split}/step_loss": loss.item(),
            f"{split}/running_avg_loss": running_avg_loss,
            "epoch": epoch,
            "step": batch_idx,
        }
    )
    return {"avg_loss": f"{running_avg_loss:.4f}"}


def log_epoch_metrics(wandb, train_loss, val_loss, epoch):
    """Log metrics for entire epoch."""
    wandb.log(
        {
            "train/epoch_loss": train_loss,
            "val/epoch_loss": val_loss,
            "epoch": epoch,
        }
    )
    print(f"Epoch {epoch+1} average train loss: {train_loss:.4f}")
    print(f"Epoch {epoch+1} average val loss: {val_loss:.4f}")


def train(
    device,
    num_heads=4,
    num_inner=1024,
    num_epochs=1,
    lr=1e-4,
    weight_decay=0.01,
    debug=True,
):
    set_seed()

    # Initialize wandb
    wandb.init(
        project="image-captioning",
        config={
            "num_heads": num_heads,
            "num_inner": num_inner,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "debug": debug,
        },
    )

    train_dataloader = get_flickr_dataloader(device, split="train")
    val_dataloader = get_flickr_dataloader(device, split="val")

    decoder = Decoder(n_head=num_heads, n_inner=num_inner).to(device)
    optimizer = optim.AdamW(decoder.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        # Training
        decoder.train()
        total_train_loss = 0  # Sum of all batch losses
        train_iter = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")

        for batch_idx, batch in enumerate(train_iter, 1):
            loss = compute_loss(batch, decoder, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            postfix = log_batch_metrics(
                wandb, loss, total_train_loss, batch_idx, epoch, "train"
            )
            train_iter.set_postfix(postfix)

            if debug and batch_idx >= debug_batch_num:
                break

        # Validation
        decoder.eval()
        total_val_loss = 0
        val_iter = tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}")

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_iter, 1):
                loss = compute_loss(batch, decoder, device)
                total_val_loss += loss.item()
                postfix = log_batch_metrics(
                    wandb, loss, total_val_loss, batch_idx, epoch, "val"
                )
                val_iter.set_postfix(postfix)

                if debug and batch_idx >= debug_batch_num:
                    break

        # Log epoch results
        average_train_loss = total_train_loss / batch_idx
        average_val_loss = total_val_loss / batch_idx
        log_epoch_metrics(wandb, average_train_loss, average_val_loss, epoch)

    wandb.finish()


if __name__ == "__main__":
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device", device)

    train(device=device, num_heads=2, num_inner=512, debug=True)
