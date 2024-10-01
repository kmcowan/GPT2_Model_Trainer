import os

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def fix_size_mismatch(model_path_or_repo, output_dir="output_directory", from_huggingface=True):
    """
    Fixes the size mismatch between model weights and the current tokenizer size.

    Args:
    - model_path_or_repo (str): Path to the local model directory or Hugging Face model repository ID.
    - output_dir (str): Directory where the corrected model and tokenizer will be saved.
    - from_huggingface (bool): If True, load the model from Hugging Face using transformers.
                               If False, load the model from the local directory using PyTorch.

    Returns:
    - None: Saves the corrected model and tokenizer to the specified output directory.
    """

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if from_huggingface:
        # Load the model and tokenizer from Hugging Face
        print(f"Loading model and tokenizer from Hugging Face repository: {model_path_or_repo}")
        model = GPT2LMHeadModel.from_pretrained(model_path_or_repo, ignore_mismatched_sizes=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path_or_repo)
    else:
        # Load the model and tokenizer from a local directory
        print(f"Loading model and tokenizer from local directory: {model_path_or_repo}")
        model = GPT2LMHeadModel.from_pretrained(model_path_or_repo, ignore_mismatched_sizes=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path_or_repo)

    # Check if the model's vocab size matches the tokenizer's vocab size
    if model.config.vocab_size != len(tokenizer):
        print(f"Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}...")

        # Resize the model's embedding layer and output layer
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)

    # Save the corrected model and tokenizer
    print(f"Saving corrected model and tokenizer to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Model weights correction completed successfully.")


if __name__ == "__main__":
    # Example usage:
    # Update these paths according to your setup
    model_path_or_repo = "kmcowan/rockymtn_gpt2"  # Replace with your Hugging Face repo ID or local path
    output_dir = "huggingface"  # Directory where the corrected model and tokenizer will be saved

    # Set to True if loading from Hugging Face, False if loading from a local directory
    from_huggingface = True

    fix_size_mismatch(model_path_or_repo, output_dir, from_huggingface)
