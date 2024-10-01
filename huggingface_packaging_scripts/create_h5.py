import h5py
import torch

# Load PyTorch state_dict
state_dict = torch.load("../huggingface_updated/pytorch_model.bin", map_location="cpu")

# Save as HDF5
with h5py.File("../huggingface_updated/tf_model.h5", "w") as h5f:
    for key, value in state_dict.items():
        h5f.create_dataset(key, data=value.cpu().numpy())
