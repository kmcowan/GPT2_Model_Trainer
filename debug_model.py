import torch
from transformers import BatchEncoding


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