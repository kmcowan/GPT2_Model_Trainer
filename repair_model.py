import torch
from transformers import GPT2LMHeadModel

def load_and_save_model(model_path, save_path):
    try:
        # Load the entire checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        # Extract the model's state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint  # If it was saved as a pure model's state_dict

        # Initialize a new GPT2 model
        model = GPT2LMHeadModel.from_pretrained('gpt2')  # Load base GPT-2 architecture

        # Load the extracted state_dict into the model
        model.load_state_dict(state_dict, strict=False)

        print("Model loaded successfully with the corrected state_dict.")

        # Save the fixed model
        torch.save(model.state_dict(), save_path)
        print(f"Model saved successfully to {save_path}")

        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Usage example
model_path = "rockymtn_gpt2.pth"  # Path to your saved model checkpoint
save_path = "huggingface_updated/fixed_rockymtn_gpt2.pth"  # Path to save the fixed model
model = load_and_save_model(model_path, save_path)