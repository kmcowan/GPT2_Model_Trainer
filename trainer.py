import torch
import torch.nn as nn  # nn is from torch's neural network module
from torch.nn import functional as F
from transformers import GPT2Tokenizer

from GPT2Model import GPT2Model
from RealTimeLossMonitor import RealTimeLossMonitor

# Hyperparameters
vocab_size = 50257  # GPT-2 vocab size
embed_dim = 256  # Embedding size (can be larger)
num_heads = 8  # Number of attention heads
num_layers = 6  # Number of transformer layers
learning_rate = 3e-4  # Learning rate

# Step 1: Tokenize the input text
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "Hello, how are you doing today?"
inputs = tokenizer(text, return_tensors='pt')



# Initialize the model
model = GPT2Model(vocab_size, embed_dim, num_heads, num_layers)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Initialize the loss monitor with a patience of 3 epochs
monitor = RealTimeLossMonitor(patience=3)

# Step 3: Example of the inputs variable and training loop
def train(model, optimizer, loss_fn, inputs, num_epochs=3, config={}):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.train()
    input_ids = inputs['input_ids']  # Tokenized input text
    target_ids = input_ids.clone()  # For simplicity, targets are the same as input
    vocab_size = tokenizer.vocab_size
    embed_dim = config.get("embed_dim", 256)
    num_heads = config.get("num_heads", 8)
    num_layers = config.get("num_layers", 6)
    learning_rate = config.get("learning_rate", 3e-4)
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        logits = model(input_ids)  # Forward pass
        loss = loss_fn(logits.view(-1, vocab_size), target_ids.view(-1))  # Compute loss

        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        '''
        continue_training = monitor.add_epoch_loss(train_loss, val_loss)
        if not continue_training:
            break  # Stop the training loop early
        '''
        # Path to save the model
        model_save_path = config.get("model_name")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f"checkpoint/model_checkpoint_epoch_{epoch}.pth")
    save_model(model, optimizer, model_save_path)

def save_model(model, optimizer, model_save_path):
    # Save the model's state_dict and the optimizer's state_dict
    print("Save model: " + model_save_path)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_save_path)

# Example of saving after training
#
# Train the model
#train(model, optimizer, loss_fn, inputs, {})
