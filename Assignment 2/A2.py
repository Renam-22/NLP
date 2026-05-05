import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

nltk.download('punkt')

docs = [
    "I love machine learning",
    "Machine learning is powerful",
    "I love python programming"
]


cv = CountVectorizer()
bow = cv.fit_transform(docs)

print("Vocabulary:", cv.get_feature_names_out())
print("Count Matrix:\n", bow.toarray())



bow_array = bow.toarray()
normalized = bow_array / bow_array.sum(axis=1, keepdims=True)
print("Normalized Count:\n", normalized)




tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(docs)

print("TF-IDF Matrix:\n", tfidf_matrix.toarray())


tokenized = [nltk.word_tokenize(doc.lower()) for doc in docs]


model = Word2Vec(sentences=tokenized, vector_size=50, window=3, min_count=1)


print("Vector for 'machine':\n", model.wv['machine'])