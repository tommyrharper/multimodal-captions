from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel

# Load dataset from Hugging Face
dataset = load_dataset("nlphuji/flickr30k", split="test", cache_dir="./data")

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class Flickr30kCLIPDataset(Dataset):
    def __init__(self, hf_dataset, processor):
        self.hf_dataset = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"]  # PIL image
        captions = item["caption"]  # List of captions
        caption = captions[0]

        # Preprocess image using CLIP processor
        inputs = self.processor(images=image, return_tensors="pt")

        # Extract pixel values (PyTorch tensor)
        pixel_values = inputs["pixel_values"].squeeze(0)  # Shape: (3, 224, 224)

        return pixel_values, caption  # Return tensor + captions

# Create dataset instance
flickr_dataset = Flickr30kCLIPDataset(dataset, processor)

# Define batch size
batch_size = 32

# Create PyTorch DataLoader
dataloader = DataLoader(flickr_dataset, batch_size=batch_size, shuffle=True)

# Fetch one batch
images, caption = next(iter(dataloader))

print("Batch Image Tensor Shape:", images.shape)  # Should be (batch_size, 3, 224, 224)
print("First image caption:", caption)  # List of caption