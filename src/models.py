import torch
from torch import nn
from transformers import GPT2Config, GPT2Model


class Decoder(nn.Module):
    def __init__(
        self,
        n_head=4,
        n_inner=1024,
        clip_embedding_dim=512,
        max_seq_length=77,
        dropout=0.1,
    ):
        super().__init__()

        # Get GPT2's config but override some params
        self.config = GPT2Config(
            n_head=n_head,  # Reduce number of attention heads
            n_inner=n_inner,  # Smaller feed-forward dimension
        )

        # Project and normalize image embedding
        self.image_projection = nn.Sequential(
            nn.Linear(clip_embedding_dim, self.config.n_embd),
            nn.LayerNorm(self.config.n_embd),
            nn.Dropout(dropout),
        )

        # Fixed size causal mask for sequence length plus image token
        self.register_buffer(
            "attn_mask", torch.tril(torch.ones(max_seq_length + 1, max_seq_length + 1))
        )

        # Use GPT2's token embedding weights
        gpt2 = GPT2Model.from_pretrained("gpt2")
        self.token_embedding = gpt2.wte

        # Add dropout after embeddings
        self.embed_dropout = nn.Dropout(dropout)

        self.self_attention = nn.MultiheadAttention(
            self.config.n_embd, self.config.n_head, batch_first=True, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(self.config.n_embd)

        self.ff = nn.Sequential(
            nn.Linear(self.config.n_embd, self.config.n_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.config.n_inner, self.config.n_embd),
        )
        self.norm3 = nn.LayerNorm(self.config.n_embd)

        # Final dropout before prediction
        self.final_dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size)

    def forward(self, image_embedding, input_ids):
        image_embedding = self.image_projection(image_embedding)
        text_embeddings = self.token_embedding(input_ids)
        text_embeddings = self.embed_dropout(text_embeddings)
        sequence = torch.cat([image_embedding.unsqueeze(1), text_embeddings], dim=1)

        # self-attention + add + norm
        attn_output, _ = self.self_attention(
            sequence, sequence, sequence, attn_mask=self.attn_mask
        )
        sequence = sequence + attn_output
        sequence = self.norm2(sequence)

        # feed-forward + add + norm
        ff_output = self.ff(sequence)
        sequence = sequence + ff_output
        sequence = self.norm3(sequence)

        # Final dropout before prediction
        sequence = self.final_dropout(sequence)

        # Only compute logits for text positions (skip image token)
        text_sequence = sequence[:, 1:, :]
        logits = self.lm_head(text_sequence)

        # Return log probabilities
        return torch.log_softmax(logits, dim=-1)


if __name__ == "__main__":
    decoder = Decoder()
    image_embedding = torch.randn(1, 512)
    input_ids = torch.randint(32, (1, 77))
    labels = torch.randint(32, (1, 77))
    print("image_embedding.shape", image_embedding.shape)
    print("input_ids.shape", input_ids.shape)
    print("labels.shape", labels.shape)
    result = decoder(image_embedding, input_ids)
    print("result.shape", result.shape)
