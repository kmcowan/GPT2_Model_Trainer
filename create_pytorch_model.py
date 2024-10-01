import os

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate_pytorch_model_bin(model_path_or_repo, output_dir="output_directory", from_huggingface=True):
    """
    Generates a pytorch_model.bin file from your model.

    Args:
    - model_path_or_repo (str): Path to the local model directory or Hugging Face model repository ID.
    - output_dir (str): Directory where the pytorch_model.bin will be saved.
    - from_huggingface (bool): If True, load the model from Hugging Face using transformers.
                               If False, load the model from the local directory using PyTorch.

    Returns:
    - None: Saves the pytorch_model.bin file to the specified output directory.
    """

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if from_huggingface:
        # Load the model from Hugging Face
        print(f"Loading model from Hugging Face repository: {model_path_or_repo}")
        model = GPT2LMHeadModel.from_pretrained(model_path_or_repo)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path_or_repo)
    else:
        # Load the model from a local directory
        print(f"Loading model from local directory: {model_path_or_repo}")
        state_dict = torch.load(os.path.join(model_path_or_repo, "fixed_rockymtn_gpt2.pth"), map_location="cpu")
        model = GPT2LMHeadModel.from_pretrained(model_path_or_repo, state_dict=state_dict)

    # Save the model weights as pytorch_model.bin
    output_path = os.path.join(output_dir, "pytorch_model.bin")
    print(f"Saving model to {output_path}...")
    torch.save(model.state_dict(), output_path)

    # Save the tokenizer files to ensure you have all the necessary components
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer files saved to {output_dir}")


if __name__ == "__main__":
    # Example usage:
    # Update these paths according to your setup
    model_path_or_repo = "kmcowan/rockymtn_gpt2"  # Replace with your Hugging Face repo ID or local path
    output_dir = "huggingface_updated"  # Directory where the pytorch_model.bin will be saved

    # Set to True if loading from Hugging Face, False if loading from a local directory
    from_huggingface = True

    generate_pytorch_model_bin(model_path_or_repo, output_dir, from_huggingface)
