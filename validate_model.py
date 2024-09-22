import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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

def validate_model(model, tokenizer, validation_data):
    for i, data in enumerate(validation_data, start=1):
        prompt = data['prompt']
        expected_output = data['expected_output']

        try:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            attention_mask = input_ids.ne(pad_token_id).long()
            output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

            print(f"Test #{i}:")
            print(f"Prompt: {prompt}")
            print(f"Expected Output: {expected_output}")
            print(f"Generated Output: {generated_text}")
            print(f"Match: {generated_text.strip() == expected_output.strip()}")
            print("-" * 50)
        except Exception as e:
            print(f"Error during validation for test #{i}: {e}")

if __name__ == "__main__":
    MODEL_PATH = "fixed_rockymtn_gpt2.pth"
    model = load_model(MODEL_PATH)

    if model:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        validation_data = [
            {
                "prompt": "Once upon a time in a land far away,",
                "expected_output": "Once upon a time in a land far away, there was a beautiful princess."
            },
            # Add more validation cases as needed
        ]

        validate_model(model, tokenizer, validation_data)