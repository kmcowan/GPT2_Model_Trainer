import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the model architecture
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load the model state dictionary from the .pth file
MODEL_PATH = "fixed_rockymtn_gpt2.pth"
try:
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error loading model '{MODEL_PATH}': {e}")
    exit(1)

def generate_text(prompt, max_length=50):
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
        output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_return_sequences=1)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error generating text: {e}")
        return None

# Test cases
tests = [
    {
        "name": "Basic Text Generation",
        "prompt": "Once upon a time in a land far away,",
    },
    # Add other test cases as needed
]

if __name__ == "__main__":
    print("Running use case tests...\n")
    # Running the tests
    for i, test in enumerate(tests, start=1):
        try:
            print(f"\nRunning test #{i}: {test['name']}")
            result = generate_text(test['prompt'])
            if result:
                print(f"Result:\n{result}")
            else:
                print("No result generated.")
        except MemoryError:
            print("MemoryError: Not enough memory to load the model.")
        except Exception as e:
            print(f"Test #{i} '{test['name']}' failed with error: {e}")

    print("\nAll tests completed.")