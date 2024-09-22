import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tokenizer import tokenizer

def validate_gpt2_model(model_path, input_text="The future of AI is", max_length=50, temperature=1.0, top_k=50):

    # Load the model and tokenizer from local paths
    #tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    model = load_model(model_path) #GPT2LMHeadModel.from_pretrained(model_path)

    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    print(f"Input IDs shape: {input_ids.shape}")

    # Perform a forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        print(f"Logits shape after forward pass: {logits.shape}")

        # Check the shape of the logits
        if logits.dim() != 3:
            raise ValueError(f"Expected logits to have 3 dimensions, but got {logits.dim()}")

        # Generate text to ensure the model works correctly
        generated_text = generate_text(model, tokenizer, input_text, max_length, temperature, top_k)
        print(f"Generated text: {generated_text}")


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


def top_k_filtering(logits, top_k=50):
    if top_k > 0:
        values, indices = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(-1).expand_as(logits)
        logits = torch.where(logits < min_values, torch.full_like(logits, -float('Inf')), logits)
    return logits

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

if __name__ == "__main__":
    # Replace these paths with the actual paths to your local model and tokenizer
    model_path = "fixed_rockymtn_gpt2.pth"


    validate_gpt2_model(model_path)