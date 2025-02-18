import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads=8, ff_dim=1024, seq_length=77):
        super().__init__()

        self.causal_mask = torch.tril(torch.ones(seq_length + 1, seq_length + 1))
        self.image_projection = nn.Linear(512, embed_dim)
        self.token_embedding = nn.Embedding(77, embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        self.lm_head = nn.Linear(embed_dim, 77)

    def forward(self, image_embedding, input_ids, labels):
        image_embedding = self.image_projection(image_embedding)

        # Get embeddings from input IDs
        text_embeddings = self.token_embedding(input_ids)

        # Prepend image embedding as first token
        sequence = torch.cat([image_embedding.unsqueeze(1), text_embeddings], dim=1)

        # Apply self-attention with causal mask
        attn_output, _ = self.self_attention(sequence, sequence, sequence, attn_mask=self.causal_mask)
        sequence = self.norm1(sequence + attn_output)
        sequence = sequence + self.ff(sequence)

        # Get logits for next token prediction
        logits = self.lm_head(sequence) # (batch_size, seq_length, vocab_size) 
        return None

        # loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits

if __name__ == "__main__":
    decoder = Decoder(embed_dim=512)
    image_embedding = torch.randn(1, 512)
    input_ids = torch.randint(32, (1, 77))
    labels = torch.randint(32, (1, 77))
    print('image_embedding.shape', image_embedding.shape)
    # print('image_embedding.shape', image_embedding)
    print('input_ids.shape', input_ids.shape)
    # print('input_ids.shape', input_ids)
    print('labels.shape', labels.shape)
    # print('labels.shape', labels)
    result = decoder(image_embedding, input_ids, labels)
    print(result)
