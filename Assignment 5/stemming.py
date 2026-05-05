import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


from nltk.corpus import wordnet as wn

# Input word
word = "good"

print("Word:", word)

synonyms = set()
for syn in wn.synsets(word):
    for lemma in syn.lemmas():
        synonyms.add(lemma.name())

print("\nSynonyms:")
print(synonyms)

antonyms = set()
for syn in wn.synsets(word):
    for lemma in syn.lemmas():
        if lemma.antonyms():
            antonyms.add(lemma.antonyms()[0].name())

print("\nAntonyms:")
print(antonyms)

hypernyms = set()
for syn in wn.synsets(word):
    for hyper in syn.hypernyms():
        hypernyms.add(hyper.name())

print("\nHypernyms:")
print(hypernyms)
