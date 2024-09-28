import os
import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
import accelerate


# Define Hugging Face model repository name
MODEL_REPO = "kmcowan/rockymtn_gpt2"  # Replace with your model repository ID
TRAINING_DATA_FOLDER = "/Users/kevincowan/training_data_2"  # Folder containing your training .txt files

# Ensure the model and tokenizer are on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load the model and tokenizer from Hugging Face
print("Loading model and tokenizer from Hugging Face...")
model = GPT2LMHeadModel.from_pretrained(MODEL_REPO).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_REPO)

# Ensure padding token exists in the tokenizer (if missing)
if tokenizer.pad_token is None:
    print("Adding a padding token to the tokenizer...")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Step 2: Load and process the training data
def load_training_data(folder_path):
    # Read all .txt files in the specified folder
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            print(f"Loading file: {filename}")
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                texts.append(file.read())
    return texts

print(f"Loading training data from {TRAINING_DATA_FOLDER}...")
training_texts = load_training_data(TRAINING_DATA_FOLDER)
print(f"Loaded {len(training_texts)} training files.")

# Create a Hugging Face dataset from training texts
dataset = Dataset.from_dict({"text": training_texts})

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

print("Tokenizing training data...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print("Tokenization complete.")

# Set up data collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # Set mlm to False since this is causal LM (not masked LM)
)

# Step 3: Define the TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",              # Output directory
    overwrite_output_dir=True,           # Overwrite the contents of the output directory
    num_train_epochs=100,                # Number of training epochs
    per_device_train_batch_size=1,       # Batch size for training (adjust based on GPU memory)
    save_steps=500,                      # Save checkpoint every 500 steps
    save_total_limit=2,                  # Limit the total number of checkpoints
    logging_dir='./logs',                # Directory for storing logs
    logging_steps=50,                    # Log every 50 steps
    report_to="none",                    # Avoid sending logs to any reporting tool
    fp16=torch.cuda.is_available(),      # Enable FP16 training if on a CUDA device
)

# Step 4: Initialize the Trainer
print("Initializing Trainer...")
trainer = Trainer(
    model=model,                         # The instantiated 🤗 Transformers model to be trained
    args=training_args,                  # Training arguments, defined above
    train_dataset=tokenized_dataset,     # Training dataset
    data_collator=data_collator          # Data collator for dynamic padding
)

# Step 5: Train the model
print("Starting training...")
trainer.train()
print("Training complete.")

# Step 6: Save the updated model locally
print("Saving updated model locally...")
model.save_pretrained("./huggingface_updated")
tokenizer.save_pretrained("./huggingface_updated")
print("Model saved locally.")

# Optional Step 7: Upload the updated model back to Hugging Face Hub
from huggingface_hub import HfApi

print("Uploading updated model to Hugging Face Hub...")
api = HfApi()
api.upload_folder(
    folder_path="./huggingface_updated",  # Path where updated model files are saved
    repo_id=MODEL_REPO,                   # Your Hugging Face model repo ID
    repo_type="model"
)
print("Model training completed and uploaded to Hugging Face Hub!")