
import os
import torch
from huggingface_hub import HfApi
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score
import numpy as np
import math
import torch
import accelerate

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# Define your Hugging Face model repository
MODEL_REPO = "kmcowan/rockymtn_gpt2"  # Replace with your model repository ID

# Define the path to your fine-tuning data
TRAINING_DATA_FOLDER = "/Users/kevincowan/training_data/fine_tuning_data"  # Directory containing your training .txt files

# Load the model and tokenizer from Hugging Face
print(f"Loading model and tokenizer from Hugging Face repository: {MODEL_REPO}...")
model = GPT2LMHeadModel.from_pretrained(MODEL_REPO)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_REPO)

# Ensure padding token exists in the tokenizer (if missing)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Ensure the model and tokenizer are on the correct device
device = torch.device("cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# Split the dataset into training and validation sets
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)

print("Tokenizing training data...")
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Set up data collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # MLM is False because this is a causal language model
)

# Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    perplexity = math.exp(eval_pred.loss)
    return {"accuracy": accuracy, "perplexity": perplexity}

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_results",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
    learning_rate=3e-5,             # Smaller learning rate for fine-tuning
    weight_decay=0.01,              # Regularization
    save_steps=500,
    save_total_limit=3,
    logging_dir='./logs',
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch", # Change to "steps" if you want more frequent evaluations
    load_best_model_at_end=True,
    fp16=False,                      # Assuming GPU supports FP16 for faster training
    max_grad_norm=1.0,              # Gradient clipping
    warmup_steps=500,               # Warmup for stable training
    report_to="none",
)


# Initialize the Trainer
trainer = Trainer(
    model=model,                         # The instantiated 🤗 Transformers model to be trained
    args=training_args,                  # Training arguments
    train_dataset=train_dataset,         # Training dataset
    eval_dataset=eval_dataset,           # Validation dataset
    data_collator=data_collator,         # Data collator for dynamic padding
    compute_metrics=compute_metrics      # Custom evaluation metrics
)

# Start training
print("Starting fine-tuning...")
trainer.train()

# Evaluate the model after training
print("Evaluating the model...")
eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")

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
model.push_to_hub(MODEL_REPO)
tokenizer.push_to_hub(MODEL_REPO)
print("Fine-tuning completed and the model is saved to Hugging Face!")
