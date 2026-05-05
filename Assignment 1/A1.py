import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer, MWETokenizer, TreebankWordTokenizer, RegexpTokenizer
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer


nltk.download('punkt')
nltk.download('wordnet')


text = "Hello! I am learning NLP, and I love machine learning 😊"




print("Whitespace:", text.split())


punct = RegexpTokenizer(r'\w+')
print("Punctuation:", punct.tokenize(text))

treebank = TreebankWordTokenizer()
print("Treebank:", treebank.tokenize(text))

tweet = TweetTokenizer()
print("Tweet:", tweet.tokenize(text))


mwe = MWETokenizer([('machine', 'learning')])
print("MWE:", mwe.tokenize(word_tokenize(text)))


words = ["running", "playing", "easily", "better"]


porter = PorterStemmer()
print("Porter:", [porter.stem(w) for w in words])


snowball = SnowballStemmer("english")
print("Snowball:", [snowball.stem(w) for w in words])


lemmatizer = WordNetLemmatizer()
print("Lemmatization:", [lemmatizer.lemmatize(w) for w in words])