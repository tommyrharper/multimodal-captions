import torch
from torch import nn
from src.dataloader import get_flickr_dataloader
from src.models import Decoder
from torch import optim

def train(num_epochs=1, lr=1e-4, weight_decay=0.01):
    # todo: pass device into dataloader and train
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("device", device)

    dataloader = get_flickr_dataloader()
    decoder = Decoder().to(device)
    optimizer = optim.AdamW(decoder.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):

        batch_num = 0
        train_loss = 0
        for batch in dataloader:
            batch_num += 1
            print("batch", batch_num)


            image_embedding = batch['image_embedding'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = decoder(image_embedding, input_ids, labels)
            print("logits", logits.shape)

            loss = nn.functional.cross_entropy(
                logits.reshape(-1, decoder.config.vocab_size), 
                labels.reshape(-1), 
                ignore_index=-100
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        # print(f"Epoch {epoch+1} loss: {train_loss / len(dataloader)}")
        print(f"Epoch {epoch+1} loss: {train_loss}")

if __name__ == "__main__":
    train()
