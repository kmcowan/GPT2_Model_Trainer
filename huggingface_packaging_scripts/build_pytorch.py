import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tokenizer import tokenizer

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

def main():
    model_path = "../fixed_rockymtn_gpt2.pth"
    save_directory = "huggingface"

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Save the model and tokenizer
    save_model_and_tokenizer(model, tokenizer, save_directory)

if __name__ == "__main__":
    main()