import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, CLIPProcessor, CLIPModel


class Flickr30kCLIPDataset(Dataset):
    def __init__(self, hf_dataset, clip_processor, clip_model, tokenizer):
        self.hf_dataset = hf_dataset
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.hf_dataset)

    def get_image_embedding(self, image):
        # Move image inputs to same device as CLIP model
        image_inputs = self.clip_processor(images=image, return_tensors="pt")
        image_inputs = {
            k: v.to(self.clip_model.device) for k, v in image_inputs.items()
        }

        with torch.no_grad():
            return self.clip_model.get_image_features(
                pixel_values=image_inputs["pixel_values"]
            ).squeeze(0)

    def get_input_ids(self, captions):
        caption = captions[torch.randint(0, len(captions), (1,)).item()]

        # Use provided tokenizer
        text_inputs = self.tokenizer(
            caption,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=77,
        )

        input_ids = text_inputs["input_ids"].squeeze(0)
        return input_ids

    def get_labels(self, input_ids):
        # Create labels (shifted input_ids for next token prediction)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]  # shift left by 1
        labels[-1] = -100  # ignore last token prediction
        return labels

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image_embeddings = self.get_image_embedding(item["image"])
        input_ids = self.get_input_ids(item["caption"])
        labels = self.get_labels(input_ids)

        return {
            "image_embedding": image_embeddings,  # (512,)
            "input_ids": input_ids,  # (77,)
            "labels": labels,  # (77,)
        }


def get_flickr_dataloader(
    device, split="train", batch_size=32, train_ratio=0.8, seed=42
):
    # Load full dataset
    full_dataset = load_dataset("nlphuji/flickr30k", split="test", cache_dir="./data")

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)

    # Split dataset
    full_dataset = full_dataset.shuffle(seed=seed)
    train_dataset = full_dataset.select(range(train_size))
    val_dataset = full_dataset.select(range(train_size, total_size))

    # Select appropriate split
    dataset = train_dataset if split == "train" else val_dataset

    # Load models and move CLIP to device
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset instance
    flickr_dataset = Flickr30kCLIPDataset(
        dataset, clip_processor, clip_model, tokenizer
    )

    # Create and return DataLoader
    return DataLoader(flickr_dataset, batch_size=batch_size, shuffle=(split == "train"))


if __name__ == "__main__":
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataloader = get_flickr_dataloader(device)
    # Fetch a batch
    batch = next(iter(dataloader))
    image_embeddings = batch["image_embedding"]  # [B, 512]
    input_ids = batch["input_ids"]  # [B, 77]
    labels = batch["labels"]  # [B, 77]

    print("Image Embeddings:", image_embeddings.shape)  # [32, 512]
    print("Input IDs:", input_ids.shape)  # [32, 77]
    print("Labels:", labels.shape)  # [32, 77]
