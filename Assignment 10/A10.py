import nltk
from nltk.util import ngrams
from collections import defaultdict, Counter


nltk.download('punkt')


text = "I love machine learning and I love coding in python"


tokens = nltk.word_tokenize(text.lower())


bigrams = list(ngrams(tokens, 2))


model = defaultdict(Counter)

for w1, w2 in bigrams:
    model[w1][w2] += 1


def autocomplete(word):
    if word in model:
        return model[word].most_common(3)
    else:
        return "No suggestion"



print("Next words after 'love':", autocomplete("love"))
print("Next words after 'machine':", autocomplete("machine"))