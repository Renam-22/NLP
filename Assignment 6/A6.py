from transformers import MarianMTModel, MarianTokenizer

# Load model for English → Hindi
model_name = "Helsinki-NLP/opus-mt-en-hi"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Input text
text = ["Hello, how are you? This is a machine translation example."]

# Tokenize
tokens = tokenizer(text, return_tensors="pt", padding=True)

# Generate translation
translated = model.generate(**tokens)

# Decode output
result = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

print("Translated Text:", result[0])