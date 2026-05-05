import nltk
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer



nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



data = {
    "text": [
        "I love Machine Learning!",
        "Python is great for NLP.",
        "I enjoy coding and learning new things."
    ],
    "label": ["positive", "positive", "neutral"]
}

df = pd.DataFrame(data)

def clean_text(text):
    text = text.lower()  
    text = text.translate(str.maketrans('', '', string.punctuation))  
    tokens = nltk.word_tokenize(text) 
    return tokens

df['tokens'] = df['text'].apply(clean_text)




stop_words = set(stopwords.words('english'))
df['tokens'] = df['tokens'].apply(lambda words: [w for w in words if w not in stop_words])




lemmatizer = WordNetLemmatizer()
df['tokens'] = df['tokens'].apply(lambda words: [lemmatizer.lemmatize(w) for w in words])

# Join tokens back to sentence
df['clean_text'] = df['tokens'].apply(lambda words: " ".join(words))




le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])




tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['clean_text'])

print("TF-IDF:\n", tfidf_matrix.toarray())



df.to_csv("cleaned_data.csv", index=False)

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
tfidf_df.to_csv("tfidf_output.csv", index=False)

print("Files saved successfully!")