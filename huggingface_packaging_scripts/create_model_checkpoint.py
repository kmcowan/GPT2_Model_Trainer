import torch
import numpy as np
import tensorflow as tf

# Load your PyTorch model state_dict
state_dict = torch.load("../huggingface/pytorch_model.bin", map_location="cpu")

# Create a new TensorFlow checkpoint
tf_checkpoint_dir = "../huggingface/tf_checkpoint"
tf_ckpt = tf.train.Checkpoint(model=tf.Module())

for key, value in state_dict.items():
    key = key.replace("transformer.", "")  # Adjust the key name as needed
    value = value.cpu().numpy()

    # Assign to TensorFlow checkpoint (add a dimension for biases if needed)
    var = tf.Variable(value, name=key)
    setattr(tf_ckpt.model, key.replace(".", "/"), var)

# Save the TensorFlow checkpoint
tf_ckpt.save(f"{tf_checkpoint_dir}/model.ckpt")
