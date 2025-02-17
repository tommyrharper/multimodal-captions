import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel

# Load dataset from Hugging Face
dataset = load_dataset("nlphuji/flickr30k", split="test", cache_dir="./data")

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)


class Flickr30kCLIPDataset(Dataset):
    def __init__(self, hf_dataset, processor, model):
        self.hf_dataset = hf_dataset
        self.processor = processor
        self.model = model

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"]
        caption = item["caption"][0]  # Select only the first caption

        # Process image & compute CLIP image embedding
        image_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"]

        with torch.no_grad():  # Freeze CLIP
            image_embedding = self.model.get_image_features(pixel_values).squeeze(0)

        # Process text with padding
        text_inputs = self.processor(
            text=caption, 
            return_tensors="pt", 
            truncation=True, 
            padding="max_length",
            max_length=77  # CLIP's max length
        )
        
        # return {
        #     'image_embedding': image_embedding,
        #     'input_ids': text_inputs["input_ids"].squeeze(0),
        #     'attention_mask': text_inputs["attention_mask"].squeeze(0)
        # }
        return image_embedding, text_inputs["input_ids"].squeeze(0), text_inputs["attention_mask"].squeeze(0)


# Create dataset instance
flickr_dataset = Flickr30kCLIPDataset(dataset, processor, model)

# Define batch size
batch_size = 32

# Create PyTorch DataLoader
dataloader = DataLoader(flickr_dataset, batch_size=batch_size, shuffle=True)

# Fetch a batch
image_embeddings, text_embeddings, thing = next(iter(dataloader))

if __name__ == "__main__":
    print("Image Embeddings Shape:", image_embeddings.shape)  # (32, 512)
    print("Text Embeddings Shape:", text_embeddings.shape)  # (32, 512)
