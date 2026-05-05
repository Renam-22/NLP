import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

text = "Natural Language Processing is fun. I love learning NLP with Python!"


sentences = sent_tokenize(text)
print("Sentences:", sentences)


words = word_tokenize(text)
print("Words:", words)


stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if w.lower() not in stop_words]
print("Filtered Words:", filtered_words)


pos_tags = nltk.pos_tag(words)
print("POS Tags:", pos_tags)


freq = FreqDist(filtered_words)
print("Word Frequency:", freq.most_common(5))