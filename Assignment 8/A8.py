import nltk
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('wordnet')

sentence = "I went to the bank to deposit money"

words = word_tokenize(sentence)

sense = lesk(words, 'bank')

print("Word:", "bank")
print("Sense:", sense)
print("Definition:", sense.definition())
print("Examples:", sense.examples())