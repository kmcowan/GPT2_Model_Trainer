import json

import torch
import transformers
from transformers import GPT2Tokenizer

from GPT2Model import GPT2Model
from GPTModel import GPTModel
# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def load_model(model_path, embed_dim, num_heads, num_layers, vocab_size, learning_rate):
    # Initialize the model with the same architecture as during training
    model = GPT2Model(vocab_size, embed_dim, num_heads, num_layers)

    # Load the saved model state dict
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set the model to evaluation mode
    model.eval()

    return model


import torch
import torch.nn.functional as F


def generate_text_2(model, tokenizer, input_text, max_length=50, temperature=1.0, top_k=50):
    model.eval()
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            print(f"Logits shape: {logits.shape}")

            logits = logits[:, -1, :] / temperature
            filtered_logits = top_k_filtering(logits, top_k=top_k)
            probabilities = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def top_k_filtering(logits, top_k=50):
    if top_k > 0:
        values, indices = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(-1).expand_as(logits)
        logits = torch.where(logits < min_values, torch.full_like(logits, -float('Inf')), logits)
    return logits

def generate_text_with_sampling(model, tokenizer, start_text="Hello", max_length=50, top_k=50):
    model.eval()
    input_ids = tokenizer(start_text, return_tensors='pt')['input_ids']  # (batch_size, sequence_length)

    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass to get the logits for the current input sequence
            logits = model(input_ids)

            # Get the logits for the last token in the sequence
            logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)

            # Apply top-k sampling to reduce the candidate tokens
            top_k_logits = torch.topk(logits, top_k).values  # Shape: (batch_size, top_k)

            # Convert logits to probabilities and sample from the distribution
            probabilities = torch.softmax(top_k_logits, dim=-1)
            next_token_id = torch.multinomial(probabilities, 1)  # Shape: (batch_size, 1)

            # Append the new token to the input_ids
            input_ids = torch.cat([input_ids, next_token_id], dim=1)  # Append the new token

    # Convert the generated token IDs back to text
    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return output_text


# Define a function to generate text using the loaded model
def generate_text(model, tokenizer, start_text="Hello", max_length=50):
    model.eval()  # Make sure the model is in evaluation mode
    input_ids = tokenizer(start_text, return_tensors='pt')['input_ids']

    # Generate text by iteratively predicting the next token
    with torch.no_grad():  # Disable gradient computation for inference
        for _ in range(max_length):
            logits = model(input_ids)  # Forward pass
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)  # Get the next token
            input_ids = torch.cat([input_ids, next_token_id], dim=1)  # Append the new token to input_ids

    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)  # Convert tokens back to text
    return output_text


import torch

def calculate_accuracy(outputs, targets):
    # Get the predicted class by finding the index with the maximum value in the output tensor
    _, predicted = torch.max(outputs, dim=1)

    # Compare the predicted classes with the actual targets
    correct = (predicted == targets).sum().item()

    # Calculate accuracy as the percentage of correct predictions
    accuracy = correct / targets.size(0)

    return accuracy * 100  # Return accuracy as a percentage


def evaluate_model(model, inputs, loss_fn):
    # Load the config.json file
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    vocab_size = tokenizer.vocab_size
    embed_dim = 256
    num_heads = 8
    num_layers = 6
    learning_rate = 3e-4
    model_save_path = config.get("model_name")
    # Load the saved model
    model = load_model(model_save_path, embed_dim, num_heads, num_layers, vocab_size, learning_rate)

    model.eval()
    with torch.no_grad():
        if isinstance(inputs, transformers.BatchEncoding):
            inputs = inputs['input_ids']
        elif not isinstance(inputs, torch.Tensor):
            raise TypeError(f"Expected input type torch.Tensor or transformers.BatchEncoding, but got {type(inputs)}")

        outputs = model(inputs)
        # Ensure the target shape matches the output shape
        targets = inputs.view(-1)
        outputs = outputs.view(-1, outputs.size(-1))

        loss = loss_fn(outputs, targets)
    accuracy = calculate_accuracy(outputs, targets)

    # Use the model to generate text
    generated_text = generate_text(model, tokenizer, start_text="The future of AI is", max_length=50)
    print("Generated Text: ", generated_text)

    # Generate text using top-k sampling
    generated_text = generate_text_with_sampling(model, tokenizer, start_text="The future of AI is", max_length=50, top_k=50)
    print(generated_text)

    generated_text = generate_text_2(model, tokenizer, "The future of AI is", max_length=50, temperature=0.7, top_k=50)
    print(generated_text)

    return loss.item(), accuracy

# Generate text using the trained model
if __name__ == "__main__":
    # load the config.json file
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    # Model hyperparameters (same as used during training)
    vocab_size = tokenizer.vocab_size
    embed_dim = 256
    num_heads = 8
    num_layers = 6
    learning_rate = 3e-4
    model_save_path = config.get("model_name")
    # Load the saved model
    model = load_model(model_save_path, embed_dim, num_heads, num_layers, vocab_size, learning_rate)

    # Assuming `inputs` and `loss_fn` are defined somewhere in your script
    inputs = tokenizer("The future of AI is", return_tensors='pt')['input_ids']  # Example input
    loss_fn = torch.nn.CrossEntropyLoss()  # Example loss function

    # Call the evaluate_model function with the additional arguments
    evaluate_model(model, inputs, loss_fn)