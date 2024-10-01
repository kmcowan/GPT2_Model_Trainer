import torch
import numpy as np
import tensorflow as tf
import h5py
import msgpack
import msgpack_numpy as m
import os

"""
This script creates all the necessary files to package a trained PyTorch model into Hugging Face compatible formats.
The following files will be created:
- `pytorch_model.bin`
- `tf_model.h5`
- TensorFlow checkpoint (`model.ckpt`)
- `flax_model.msgpack`

"""
# Enable NumPy support for msgpack
m.patch()


def package_huggingface_model(model, save_dir="huggingface"):
    """
    Packages a trained PyTorch model into Hugging Face compatible formats:
    - pytorch_model.bin
    - tf_model.h5
    - TensorFlow checkpoint
    - flax_model.msgpack

    Args:
    model (torch.nn.Module): Trained PyTorch model.
    save_dir (str): Directory to save all the files.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Step 1: Save PyTorch model as `pytorch_model.bin`
    print("Saving PyTorch model as pytorch_model.bin...")
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

    # Load the state_dict
    state_dict = model.state_dict()

    # Step 2: Create TensorFlow Checkpoint (`model.ckpt`)
    print("Converting to TensorFlow checkpoint...")
    tf_checkpoint_dir = os.path.join(save_dir, "tf_checkpoint")
    os.makedirs(tf_checkpoint_dir, exist_ok=True)
    tf_ckpt = tf.train.Checkpoint(model=tf.Module())

    for key, value in state_dict.items():
        key = key.replace("transformer.", "")  # Adjust the key name as needed
        value = value.cpu().numpy()

        # Assign to TensorFlow checkpoint (add a dimension for biases if needed)
        var = tf.Variable(value, name=key)
        setattr(tf_ckpt.model, key.replace(".", "/"), var)

    # Save the TensorFlow checkpoint
    tf_ckpt.save(os.path.join(tf_checkpoint_dir, "model.ckpt"))
    print(f"TensorFlow checkpoint saved to {tf_checkpoint_dir}")

    # Step 3: Save the model as an HDF5 file (`tf_model.h5`)
    print("Saving model as tf_model.h5...")
    with h5py.File(os.path.join(save_dir, "tf_model.h5"), "w") as h5f:
        for key, value in state_dict.items():
            h5f.create_dataset(key, data=value.cpu().numpy())
    print(f"tf_model.h5 saved in {save_dir}")

    # Step 4: Save as Flax model (`flax_model.msgpack`)
    print("Saving model as flax_model.msgpack...")
    flax_params = {key: value.cpu().numpy() for key, value in state_dict.items()}
    with open(os.path.join(save_dir, "flax_model.msgpack"), "wb") as f:
        packed = msgpack.packb(flax_params)
        f.write(packed)
    print(f"flax_model.msgpack saved in {save_dir}")

    print("Packaging completed successfully.")

def save_model_and_tokenizer(model, tokenizer, save_directory):
    """
    Save the model and tokenizer to the specified directory.
    """
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved to {save_directory}")

def load_model_and_tokenizer(model_path):
    """
    Load the model and tokenizer from the specified path.
    """
    #model_path = "rockymtn_gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = load_model(model_path)
    print(f"Model and tokenizer loaded from {model_path}")
    return model, tokenizer

def load_model(model_path):
    try:
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Model '{model_path}' loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model '{model_path}': {e}")
        return None

if __name__ == "__main__":
    # Example usage
    from transformers import GPT2LMHeadModel, GPT2Tokenizer  # Adjust if you're using a different architecture

    model_path = "huggingface_updated/fixed_rockymtn_gpt2.pth"
    save_directory = "huggingface"

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Save the model and tokenizer
    save_model_and_tokenizer(model, tokenizer, save_directory)  # Change path accordingly

    # Package the model
    package_huggingface_model(model, save_dir=save_directory)
