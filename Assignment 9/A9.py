from transformers import pipeline

# Load sentiment analysis pipeline (multilingual model)
sentiment = pipeline("sentiment-analysis",
                     model="nlptown/bert-base-multilingual-uncased-sentiment")

# Example Hindi text
texts = [
    "यह बहुत अच्छा है",
    "मुझे यह पसंद नहीं है",
    "यह ठीक है"
]

# Predict sentiment
results = sentiment(texts)

# Print results
for text, res in zip(texts, results):
    print("Text:", text)
    print("Sentiment:", res['label'], "| Score:", res['score'])
    print()