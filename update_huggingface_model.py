import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Define Hugging Face model repository name
MODEL_REPO = "kmcowan/rockymtn_gpt2"  # Replace with your model repository ID

# Step 1: Load the model and tokenizer from Hugging Face
print("Loading model and tokenizer from Hugging Face...")
model = GPT2LMHeadModel.from_pretrained(MODEL_REPO)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_REPO)

# Ensure the model and tokenizer are on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 2: Load your dataset (using a sample dataset here, replace with your actual data)
# You can use `load_dataset` or any custom data loading function
print("Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Step 3: Define the TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",              # Output directory
    overwrite_output_dir=True,           # Overwrite the contents of the output directory
    num_train_epochs=3,                  # Number of training epochs
    per_device_train_batch_size=4,       # Batch size for training
    save_steps=500,                      # Save checkpoint every 500 steps
    save_total_limit=2,                  # Limit the total number of checkpoints
    logging_dir='./logs',                # Directory for storing logs
    logging_steps=50,                    # Log every 50 steps
    report_to="none",                    # Avoid sending logs to any reporting tool
)

# Step 4: Initialize the Trainer
print("Initializing Trainer...")
trainer = Trainer(
    model=model,                         # The instantiated 🤗 Transformers model to be trained
    args=training_args,                  # Training arguments, defined above
    train_dataset=tokenized_dataset,     # Training dataset
)

# Step 5: Train the model
print("Starting training...")
trainer.train()

# Step 6: Save the updated model locally
print("Saving updated model locally...")
model.save_pretrained("./huggingface")
tokenizer.save_pretrained("./huggingface")

# Optional Step 7: Upload the updated model back to Hugging Face Hub
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./huggingface",  # Path where updated model files are saved
    repo_id=MODEL_REPO,           # Your Hugging Face model repo ID
    repo_type="model"
)

print("Model training completed and uploaded to Hugging Face Hub!")
