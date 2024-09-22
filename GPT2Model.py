import torch
import torch.nn as nn
from transformers import BatchEncoding


class GPT2Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len=1024):
        super(GPT2Model, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.max_len = max_len
        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(max_len, embed_dim),
                                                requires_grad=False)

        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def _generate_positional_encoding(self, max_len, embed_dim):
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        if isinstance(x, BatchEncoding):
            x = x['input_ids']
        elif not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input type torch.Tensor or transformers.BatchEncoding, but got {type(x)}")

        print(f"Input type: {type(x)}, shape: {x.shape}")

        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")

        pos_encoding = self.positional_encoding[:, :seq_len, :]
        print(f"Positional encoding shape: {pos_encoding.shape}")

        x = self.embed(x) + pos_encoding
        print(f"Shape after adding embedding and positional encoding: {x.shape}")

        for i, transformer in enumerate(self.transformer_blocks):
            x = transformer(x)
            print(f"Shape after transformer block {i}: {x.shape}")

        logits = self.fc_out(x)
        print(f"Logits shape: {logits.shape}")

        return logits
''' 
    def forward(self, x):
        if isinstance(x, BatchEncoding):
            x = x['input_ids']
        elif not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input type torch.Tensor or transformers.BatchEncoding, but got {type(x)}")

        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")

        # Slice positional encoding dynamically based on the input length
        pos_encoding = self.positional_encoding[:, :seq_len, :]

        # Add embedding and positional encoding
        x = self.embed(x) + pos_encoding

        for transformer in self.transformer_blocks:
            x = transformer(x)

        logits = self.fc_out(x)
        return logits
        '''
