import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, CLIPProcessor, CLIPModel

class Flickr30kCLIPDataset(Dataset):
    def __init__(self, hf_dataset, clip_processor, clip_model):
        self.hf_dataset = hf_dataset
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        # Add GPT2 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Add image projection layer to match decoder dimensions if needed
        # self.image_projection = nn.Linear(512, decoder_hidden_size)

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"]
        caption = item["caption"][0]

        # Get image embedding
        image_inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_embedding = self.clip_model.get_image_features(pixel_values=image_inputs["pixel_values"]).squeeze(0)

        # Use GPT2 tokenizer instead of CLIP
        text_inputs = self.tokenizer(
            caption,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=77
        )
        
        input_ids = text_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs["attention_mask"].squeeze(0)
        
        # Create labels (shifted input_ids for next token prediction)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]  # shift left by 1
        labels[-1] = -100  # ignore last token prediction
        
        return {
            'image_embedding': image_embedding,  # (512,)
            'input_ids': input_ids,  # (77,)
            'attention_mask': attention_mask,  # (77,)
            'labels': labels  # (77,)
        }

def get_flickr_dataloader(split="train", batch_size=32):
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

    # Create dataset instance
    flickr_dataset = Flickr30kCLIPDataset(dataset, processor, model)

    # Create and return DataLoader
    return DataLoader(flickr_dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    dataloader = get_flickr_dataloader()
    # Fetch a batch
    batch = next(iter(dataloader))
    image_embeddings = batch['image_embedding']  # [B, 512]
    input_ids = batch['input_ids']  # [B, 77]
    attention_mask = batch['attention_mask']  # [B, 77]
    labels = batch['labels']  # [B, 77]

    print("Image Embeddings:", image_embeddings.shape)  # [32, 512]
    print("Input IDs:", input_ids.shape)  # [32, 77]
    print("Attention Mask:", attention_mask.shape)  # [32, 77]
    print("Labels:", labels.shape)  # [32, 77]
    print("attention_mask:", attention_mask[0])  # [START, word1, word2, word3, END]

