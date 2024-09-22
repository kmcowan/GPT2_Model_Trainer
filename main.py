import json
import os
import torch
from torch.optim import Adam

from GPT2Model import GPT2Model
from GPTModel import GPTModel
import tokenizer
import data_pull
import trainer
from RealTimeLossMonitor import RealTimeLossMonitor
from evaluator import evaluate_model
from tokenizer import tokenizer
import transformers


def load_config(config_path='config.json'):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def fetch_data_from_source(config):
    data_source_type = config.get("data_source_type")
    if data_source_type == 'solr':
        solr_url = config.get("solr_url")
        query = config.get("query", "*:*")
        source_info = {'solr_url': solr_url, 'query': query}
    elif data_source_type == 'file':
        file_path = config.get("file_path")
        source_info = file_path
    elif data_source_type == 'url':
        url = config.get("url")
        source_info = url
    else:
        raise ValueError("Invalid source type! Must be 'solr', 'file', or 'url'.")

    batch_size = config.get("batch_size", 1000)
    return [data_pull.fetch_data(data_source_type, source_info, batch_size)]

def fetch_data_from_directory(directory_path):
    all_texts = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                all_texts.append(file.read())
    return all_texts

def tokenize_text(training_text):
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(training_text, return_tensors='pt', padding=True, truncation=True, max_length=1000)

def initialize_model(config, vocab_size):
    embed_dim = config.get("embed_dim", 256)
    num_heads = config.get("num_heads", 8)
    num_layers = config.get("num_layers", 6)
    learning_rate = config.get("learning_rate", 3e-4)
    max_len = 1024  # Define the maximum length for positional encoding

    model = GPT2Model(vocab_size, embed_dim, num_heads, num_layers, max_len)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    model_save_path = config.get("model_name")
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, loss_fn

def print_model_info(model, optimizer, loss_fn, inputs):
    num_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = num_parameters * 4 / (1024 ** 2)  # Assuming 4 bytes per parameter, in MB

    loss, accuracy = evaluate_model(model, inputs, loss_fn)

    print("Model Information:")
    print(f"Total parameters: {num_parameters}")
    print(f"Trainable parameters: {trainable_parameters}")
    print(f"Model size: {model_size:.2f} MB")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")

def main():
    config = load_config()

    if config.get("use_dir"):
        directory_path = config.get("directory_path")
        training_text = fetch_data_from_directory(directory_path)
    else:
        training_text = fetch_data_from_source(config)
    print("Training with " + str(len(training_text)) + " documents...")
    for text in training_text:
        inputs = tokenize_text(text)
        vocab_size = tokenizer.vocab_size
        model, optimizer, loss_fn = initialize_model(config, vocab_size)

        num_epochs = config.get("num_epochs", 3)
        trainer.train(model, optimizer, loss_fn, inputs, num_epochs, config)

    print("Training complete!")
    evaluate_model(model, inputs, loss_fn)
    print_model_info(model, optimizer, loss_fn, inputs)

if __name__ == "__main__":
    main()