import os
import torch
from huggingface_hub import HfApi
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset

# Define your Hugging Face model repository
MODEL_REPO = "kmcowan/rockymtn_gpt2"  # Replace with your model repository ID

# Define the path to your fine-tuning data
TRAINING_DATA_FOLDER = "fine_tuning_data"  # Directory containing your training .txt files

# Load the model and tokenizer from Hugging Face
print(f"Loading model and tokenizer from Hugging Face repository: {MODEL_REPO}...")
model = GPT2LMHeadModel.from_pretrained(MODEL_REPO)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_REPO)

# Ensure padding token exists in the tokenizer (if missing)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Ensure the model and tokenizer are on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to load and process the training data
def load_training_data(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                texts.append(file.read())
    return texts

print(f"Loading training data from {TRAINING_DATA_FOLDER}...")
training_texts = load_training_data(TRAINING_DATA_FOLDER)

# Create a Hugging Face Dataset from training texts
dataset = Dataset.from_dict({"text": training_texts})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)

print("Tokenizing training data...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Set up data collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # MLM is False because this is a causal language model
)

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_results",    # Directory for the output
    overwrite_output_dir=True,           # Overwrite the output directory if it exists
    num_train_epochs=5,                  # Adjust number of epochs for fine-tuning
    per_device_train_batch_size=2,       # Adjust batch size based on GPU memory
    save_steps=500,                      # Save checkpoint every 500 steps
    save_total_limit=3,                  # Limit the total number of checkpoints
    logging_dir='./logs',                # Directory for storing logs
    logging_steps=50,                    # Log every 50 steps
    report_to="none",                    # Avoid sending logs to any reporting tool
    fp16=torch.cuda.is_available(),      # Enable FP16 training for faster training if using GPU
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # The instantiated 🤗 Transformers model to be trained
    args=training_args,                  # Training arguments
    train_dataset=tokenized_dataset,     # Training dataset
    data_collator=data_collator          # Data collator for dynamic padding
)

# Start training
print("Starting fine-tuning...")
trainer.train()

# Save the fine-tuned model
fine_tuned_model_dir = "./fine_tuned_model"
print(f"Saving the fine-tuned model to {fine_tuned_model_dir}...")
model.save_pretrained(fine_tuned_model_dir)
tokenizer.save_pretrained(fine_tuned_model_dir)

# Optional: Upload the fine-tuned model to Hugging Face Hub
api = HfApi()
api.upload_folder(
    folder_path=fine_tuned_model_dir,
    repo_id=MODEL_REPO,
    repo_type="model"
)

print("Fine-tuning completed and the model is saved to Hugging Face!")
