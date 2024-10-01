import torch
import numpy as np
from transformers import GPT2LMHeadModel

# Define paths
MODEL_PATH = "huggingface_updated/pytorch_model.bin"  # Update this path
OUTPUT_PATH = "huggingface_updated/pytorch_model.bin"

# Load the model's state_dict
print("Loading model weights...")
state_dict = torch.load(MODEL_PATH, map_location="cpu")


def correct_weights(state_dict):
    """
    Detects and corrects NaN or inf values in model weights.

    Args:
    state_dict (dict): The state dictionary containing model weights.

    Returns:
    dict: Corrected state dictionary.
    """
    corrected_state_dict = {}
    corrected_weights = 0

    for key, param in state_dict.items():
        if torch.is_tensor(param):
            # Convert to NumPy array for easier handling of NaNs and Infs
            param_np = param.numpy()

            # Check for NaNs or Infs
            if np.isnan(param_np).any() or np.isinf(param_np).any():
                print(f"Correcting weights for tensor: {key}")

                # Replace NaNs and Infs with 0 or small random values
                param_np = np.nan_to_num(param_np, nan=0.0, posinf=0.0, neginf=0.0)

                # Increment corrected weights count
                corrected_weights += 1

            # Convert back to tensor
            corrected_state_dict[key] = torch.from_numpy(param_np)
        else:
            # Directly copy non-tensor values
            corrected_state_dict[key] = param

    print(f"Total tensors corrected: {corrected_weights}")
    return corrected_state_dict


# Correct the weights
print("Correcting model weights...")
corrected_state_dict = correct_weights(state_dict)

# Save the corrected state_dict
print(f"Saving corrected weights to {OUTPUT_PATH}...")
torch.save(corrected_state_dict, OUTPUT_PATH)

print("Model weights correction completed.")
