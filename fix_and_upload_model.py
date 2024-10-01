import torch
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from huggingface_hub import HfApi
import numpy as np

# Step 1: Define your Hugging Face model repository name
MODEL_REPO = "kmcowan/rockymtn_gpt2"  # Replace with your model repository ID
FIXED_MODEL_REPO = "kmcowan/rockymtn_gpt2_fixed"  # Name of the new repo to save the fixed model

# Step 1: Load the model and tokenizer from Hugging Face
print(f"Loading model and tokenizer from Hugging Face repository: {MODEL_REPO}...")
model = GPT2LMHeadModel.from_pretrained(MODEL_REPO, ignore_mismatched_sizes=True)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_REPO)

# Ensure padding token exists in the tokenizer (if missing)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


def fix_model_weights(state_dict):
    """
    Detects and fixes NaN or inf values in the model weights.

    Args:
    - state_dict (dict): The state dictionary containing model weights.

    Returns:
    - dict: The corrected state dictionary.
    """
    corrected_state_dict = {}
    corrected_weights = 0

    for key, param in state_dict.items():
        if torch.is_tensor(param):
            # Convert to NumPy array for easier handling of NaNs and Infs
            param_np = param.cpu().numpy()

            # Check for NaNs or Infs and replace them
            if np.isnan(param_np).any() or np.isinf(param_np).any():
                print(f"Correcting weights for tensor: {key}")

                # Replace NaNs and Infs with 0
                param_np = np.nan_to_num(param_np, nan=0.0, posinf=0.0, neginf=0.0)

                # Increment the corrected weights count
                corrected_weights += 1

            # Convert back to a tensor
            corrected_state_dict[key] = torch.from_numpy(param_np)
        else:
            # Copy non-tensor values as is
            corrected_state_dict[key] = param

    print(f"Total tensors corrected: {corrected_weights}")
    return corrected_state_dict


# Step 2: Fix the model weights
print("Correcting model weights...")
fixed_state_dict = fix_model_weights(model.state_dict())
model.load_state_dict(fixed_state_dict)

# Save the corrected model locally
output_dir = "./fixed_model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Saving corrected model to {output_dir}...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Step 3: Upload the corrected model back to Hugging Face
print("Uploading the corrected model to Hugging Face...")
api = HfApi()
api.upload_folder(
    folder_path=output_dir,
    repo_id=FIXED_MODEL_REPO,  # You can upload to a new repo or the original one
    repo_type="model"
)

print("Model upload completed. The corrected model is now available on Hugging Face.")
