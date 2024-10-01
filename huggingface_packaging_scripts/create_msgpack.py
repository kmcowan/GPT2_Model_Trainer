import msgpack
import msgpack_numpy as m
import torch
# Enable NumPy support for msgpack
m.patch()

# Load the PyTorch model's state_dict
state_dict = torch.load("../huggingface/pytorch_model.bin", map_location="cpu")

# Convert to NumPy arrays
flax_params = {key: value.cpu().numpy() for key, value in state_dict.items()}

# Save as msgpack
with open("../huggingface/flax_model.msgpack", "wb") as f:
    packed = msgpack.packb(flax_params)
    f.write(packed)
