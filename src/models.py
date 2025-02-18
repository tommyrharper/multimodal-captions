import torch
from torch import nn
from transformers import GPT2Config, GPT2Model


class Decoder(nn.Module):
    def __init__(self, n_head=4, n_inner=1024):
        super().__init__()

        # Get GPT2's config but override some params
        self.config = GPT2Config(
            n_head=n_head,  # Reduce number of attention heads
            n_inner=n_inner,  # Smaller feed-forward dimension
        )

        self.image_projection = nn.Linear(512, self.config.n_embd)

        # Fixed size causal mask for our specific sequence length (77 tokens + 1 image token)
        # note this is effectively saved as self.attn_mask
        self.register_buffer("attn_mask", torch.tril(torch.ones(78, 78)))

        # Use GPT2's token embedding weights
        gpt2 = GPT2Model.from_pretrained("gpt2")
        self.token_embedding = gpt2.wte

        self.self_attention = nn.MultiheadAttention(
            self.config.n_embd, self.config.n_head, batch_first=True
        )
        self.norm1 = nn.LayerNorm(self.config.n_embd)
        self.ff = nn.Sequential(
            nn.Linear(self.config.n_embd, self.config.n_inner),
            nn.ReLU(),
            nn.Linear(self.config.n_inner, self.config.n_embd),
        )
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size)

    def forward(self, image_embedding, input_ids, labels):
        image_embedding = self.image_projection(image_embedding)
        text_embeddings = self.token_embedding(input_ids)
        sequence = torch.cat([image_embedding.unsqueeze(1), text_embeddings], dim=1)
        attn_output, _ = self.self_attention(
            sequence, sequence, sequence, attn_mask=self.attn_mask
        )
        sequence = self.norm1(sequence + attn_output)
        sequence = sequence + self.ff(sequence)

        # Only compute logits for text positions (skip image token)
        text_sequence = sequence[:, 1:, :]
        logits = self.lm_head(text_sequence)

        return logits
        return None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            return loss


if __name__ == "__main__":
    decoder = Decoder()
    image_embedding = torch.randn(1, 512)
    input_ids = torch.randint(32, (1, 77))
    labels = torch.randint(32, (1, 77))
    print("image_embedding.shape", image_embedding.shape)
    print("input_ids.shape", input_ids.shape)
    print("labels.shape", labels.shape)
    result = decoder(image_embedding, input_ids, labels)
    print("result.shape", result.shape)
