import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import analyze_model_shape
from analyze_model_shape import validate_gpt2_model
from tokenizer import tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
''' 
# Define Hugging Face model repository name
MODEL_REPO = "kmcowan/rockymtn_gpt2"  # Replace with your model repository ID

# Ensure the model and tokenizer are on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer from Hugging Face
print("Loading model and tokenizer from Hugging Face...")
model = GPT2LMHeadModel.from_pretrained(MODEL_REPO).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_REPO)

# Ensure padding token exists in the tokenizer (if missing)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
'''

def get_chat_response(model, input_text="The future of AI is", max_length=100, temperature=1.0, top_k=50):

    # Load the model and tokenizer from local paths
    #tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    #model = load_model(model_path) #GPT2LMHeadModel.from_pretrained(model_path)

    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    #print(f"Input IDs shape: {input_ids.shape}")

    # Perform a forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        #print(f"Logits shape after forward pass: {logits.shape}")

        # Check the shape of the logits
        if logits.dim() != 3:
            raise ValueError(f"Expected logits to have 3 dimensions, but got {logits.dim()}")

        # Generate text to ensure the model works correctly
        generated_text = generate_text(model, tokenizer, input_text, max_length, temperature, top_k)
        #print(f"Generated text: {generated_text}")
        return generated_text


def generate_text(model, tokenizer, input_text, max_length=50, temperature=1.0, top_k=50):
    model.eval()
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            logits = logits[:, -1, :] / temperature
            filtered_logits = top_k_filtering(logits, top_k=top_k)
            probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


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

def top_k_filtering(logits, top_k=50, filter_value=-float('Inf')):
    """Apply top-k filtering to logits"""
    if top_k > 0:
        values, indices = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(-1).expand_as(logits)
        logits = torch.where(logits < min_values, torch.full_like(logits, -float('Inf')), logits)
    return logits


# Updated generate_response function with error handling
def generate_response(model, prompt, max_length=50, temperature=1.0, top_k=50):
    model_path = "/Users/kevincowan/huggingface/rockymtn_gpt2/rockymtn_gpt2.pth"
    #return validate_gpt2_model(model_path, input_text=prompt)

    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

            # Apply temperature scaling
            logits = logits[:, -1, :] / temperature

            # Clamp the logits to avoid extreme values
            logits = torch.clamp(logits, min=-1e10, max=1e10)

            # Apply top-k filtering
            filtered_logits = top_k_filtering(logits, top_k=top_k)

            # Handle NaN and Inf values by replacing them with 0
            filtered_logits = torch.nan_to_num(filtered_logits, nan=0.0, posinf=0.0, neginf=0.0)

            # Sample from the filtered distribution
            probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)

            # Ensure probabilities are valid before sampling
            probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)

            next_token = torch.multinomial(probabilities, 1)

            # Append the predicted token to the input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop generating if the EOS token is encountered
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the generated sequence
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

model_path = "/Users/kevincowan/huggingface/rockymtn_gpt2/rockymtn_gpt2.pth"
model = load_model(model_path)
# Chat loop
print("Welcome to the chat client! Type 'exit' to end the chat.")
history = []

while True:
    # Get user input
    user_input = input("You: ")

    if user_input.lower() == "exit" or user_input.lower() == "bye":
        print("Chat session ended. Goodbye!")
        break

    # Add user input to history
    history.append(f"You: {user_input}")

    # Construct the conversation context
    prompt = "\n".join(history) + "\nAI:"

    # Generate a response
    ai_response = get_chat_response(model, user_input)#generate_response(model, prompt)

    # Display the AI's response
    print(f"AI: {ai_response}")

    # Add AI response to history
    history.append(f"AI: {ai_response}")

    # Keep only the most recent context (e.g., last 10 turns)
    if len(history) > 20:
        history = history[-20:]
