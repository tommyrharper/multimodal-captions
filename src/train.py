import torch
from torch import nn
from src.dataloader import get_flickr_dataloader
from src.models import Decoder
from torch import optim
import random
import numpy as np

debug_batch_num = 3


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(
    device,
    num_heads=4,
    num_inner=1024,
    num_epochs=1,
    lr=1e-4,
    weight_decay=0.01,
    debug=True,
):
    set_seed()  # Set seeds before training

    train_dataloader = get_flickr_dataloader(device, split="train")
    val_dataloader = get_flickr_dataloader(device, split="val")

    decoder = Decoder(n_head=num_heads, n_inner=num_inner).to(device)
    optimizer = optim.AdamW(decoder.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        decoder.train()
        batch_num = 0
        train_loss = 0
        for batch in train_dataloader:
            batch_num += 1
            print("batch", batch_num)

            # Ensure all tensors are on the correct device
            image_embedding = batch["image_embedding"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = decoder(image_embedding, input_ids, labels)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, decoder.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if debug and batch_num >= debug_batch_num:  # Early break if in debug mode
                break

        decoder.eval()
        val_loss = 0
        val_batch_num = 0
        with torch.no_grad():
            for batch in val_dataloader:
                val_batch_num += 1
                image_embedding = batch["image_embedding"].to(device)
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                logits = decoder(image_embedding, input_ids, labels)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, decoder.config.vocab_size),
                    labels.reshape(-1),
                    ignore_index=-100,
                )

                val_loss += loss.item()

                if (
                    debug and val_batch_num >= debug_batch_num
                ):  # Early break if in debug mode
                    break

        average_train_loss = train_loss / batch_num
        average_val_loss = val_loss / batch_num
        print(f"Epoch {epoch+1} average train loss: {average_train_loss}")
        print(f"Epoch {epoch+1} average val loss: {average_val_loss}")


if __name__ == "__main__":
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device", device)

    train(device=device, num_heads=2, num_inner=512, debug=True)
