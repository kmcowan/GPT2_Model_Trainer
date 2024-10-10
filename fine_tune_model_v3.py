import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.utils.rnn import pad_sequence
import os
import logging
from huggingface_hub import HfApi, HfFolder, Repository

# Setup verbose logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to analyze the model architecture
def analyze_model(model):
    logger.info(f"Analyzing the model: {model.__class__.__name__}")
    logger.info(f"Number of layers: {len(model.transformer.h)}")
    logger.info(f"Number of parameters: {model.num_parameters()}")
    logger.info(f"Model is using {'cuda' if torch.cuda.is_available() else 'cpu'}")
    return {
        'model_name': model.__class__.__name__,
        'num_layers': len(model.transformer.h),
        'num_params': model.num_parameters(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

# Custom Dataset class for a directory of text files
class TextDataset(Dataset):
    def __init__(self, tokenizer, directory_path, block_size=512):
        self.examples = []
        files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.txt')]
        for file_path in files:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                tokens = tokenizer.encode(line.strip(), truncation=True, max_length=block_size)
                if len(tokens) > 0:  # Skip empty lines
                    self.examples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

# Collate function to pad sequences to the same length in a batch
def collate_fn(batch):
    return pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)

# Load the tokenizer and model
model_name = "kmcowan/rockymtn_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

output_dir = "./fine_tuned_model_v3a"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.path.exists(output_dir):
    model = GPT2LMHeadModel.from_pretrained(output_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
    model = model.to(device)
    logger.info("Resuming from last checkpoint...")

# Analyze the model before training
model_info = analyze_model(model)
logger.info(f"Model details: {model_info}")

# Fine-tuning parameters
train_directory = '/Users/kevincowan/training_data/fine_tuning_data'  # Change this to your directory path
epochs = 3
batch_size = 16
learning_rate = 3e-5
block_size = 512
max_grad_norm = 1.0
save_steps = 500
output_dir = './fine_tuned_model_v3a'

# Dataset and DataLoader
train_dataset = TextDataset(tokenizer, train_directory, block_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)

# Use AMP for mixed precision training
scaler = GradScaler()

model = model.to(device)
model.train()

# Training loop with verbose logging
logger.info("Starting training...")
for epoch in range(epochs):
    logger.info(f"Epoch {epoch + 1}/{epochs}")
    epoch_loss = 0
    for step, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Use correct autocast for torch 1.10+
        with autocast(device_type=device.type if torch.cuda.is_available() else 'cpu'):
            outputs = model(batch, labels=batch)
            loss = outputs.loss

        # Backward pass with AMP
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Step optimizer and scheduler
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_loss += loss.item()

        # Detailed logging
        logger.info(f"Step {step}/{len(train_loader)}: Loss = {loss.item():.4f}")

        if step % save_steps == 0 and step > 0:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Model saved at step {step}. Checkpoint loss: {loss.item():.4f}")

    logger.info(f"Epoch {epoch + 1}/{epochs} completed with average loss: {epoch_loss / len(train_loader):.4f}")

# Final save
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
logger.info(f"Training complete. Model saved to {output_dir}")

# Upload the model to Hugging Face Hub
def upload_to_huggingface(model_dir, repo_id, commit_message="Upload fine-tuned model"):
    api = HfApi()
    token = HfFolder.get_token()  # Ensure you're logged in with 'huggingface-cli login'
    repo = Repository(local_dir=model_dir, clone_from=repo_id, use_auth_token=token)
    repo.git_add(auto_lfs_track=True)  # Adds model files, uses git-lfs for large files like pytorch_model.bin
    repo.git_commit(commit_message)
    repo.git_push()

# Upload the fine-tuned model
upload_to_huggingface(output_dir, "kmcowan/rockymtn_gpt2")
logger.info("Model uploaded to Hugging Face.")
