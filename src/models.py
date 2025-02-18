import torch
from torch import nn
from transformers import GPT2Config, GPT2Model

class Decoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, ff_dim=1024, seq_length=77):
        super().__init__()
        
        # Get GPT2's vocab size and embedding dimension
        gpt2_config = GPT2Config()
        self.vocab_size = gpt2_config.vocab_size
        
        self.causal_mask = torch.tril(torch.ones(seq_length + 1, seq_length + 1))
        self.image_projection = nn.Linear(512, embed_dim)
        
        # Use GPT2's token embedding weights
        gpt2 = GPT2Model.from_pretrained('gpt2')
        self.token_embedding = gpt2.wte
        
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.lm_head = nn.Linear(embed_dim, self.vocab_size)

    def forward(self, image_embedding, input_ids, labels):
        print('----- forward pass -----')
        image_embedding = self.image_projection(image_embedding)
        print('image_embedding.shape', image_embedding.shape)
        text_embeddings = self.token_embedding(input_ids)
        print('text_embeddings.shape', text_embeddings.shape)
        sequence = torch.cat([image_embedding.unsqueeze(1), text_embeddings], dim=1)
        print('sequence.shape', sequence.shape)
        attn_output, _ = self.self_attention(sequence, sequence, sequence, attn_mask=self.causal_mask)
        print('attn_output.shape', attn_output.shape)
        sequence = self.norm1(sequence + attn_output)
        print('sequence.shape', sequence.shape)
        sequence = sequence + self.ff(sequence)
        print('sequence.shape', sequence.shape)
        
        logits = self.lm_head(sequence)

        return logits        
        return None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1), ignore_index=-100)
            return loss
            

if __name__ == "__main__":
    decoder = Decoder(embed_dim=768)
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
    print('result.shape', result.shape)
