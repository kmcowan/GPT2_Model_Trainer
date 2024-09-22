from transformers import GPT2Tokenizer

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Example text
text = "Hello, how are you doing today?"

# Tokenize the text
inputs = tokenizer(text, return_tensors='pt')

print("Token IDs:", inputs['input_ids'])
