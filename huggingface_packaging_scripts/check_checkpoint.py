import torch

checkpoint_path = "../rockymtn_gpt2.pth"
checkpoint = torch.load(checkpoint_path)

# Display the keys available in the checkpoint
print(checkpoint.keys())