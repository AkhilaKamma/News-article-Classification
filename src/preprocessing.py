
import pandas as pd
import string,nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.util import ngrams
from nltk.stem.porter import *
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")


ps = PorterStemmer()
remove_punctuation = dict((ord(char), None) for char in string.punctuation)

#Function to perform preprocessing
def get_processed_tokens(text):
    lowers = text.lower()
    no_punctuation = lowers.translate(remove_punctuation)
    tokens = nltk.word_tokenize(no_punctuation)
    filtered = [word for word in tokens if not word in stopwords.words('english')]
    lemmatize= []
    for item in filtered:
        lemmatize.append(ps.stem(item))
    return lemmatize

# Function to generate bigrams for a given text
def generate_bigrams(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Generate bigrams
    bigrams = list(ngrams(tokens, 2))
    return bigrams

#

# Function to get the keywords
def get_keywords(text):
  doc = nlp(text)
  keywords = [token.text for token in doc if token.pos_ in ["NOUN", "ADJ"]]
  return keywords

#
def get_processed_words_gb(text):
    lowers = text.lower()
    no_punctuation = lowers.translate(remove_punctuation)
    tokens = nltk.word_tokenize(no_punctuation)
    filtered = [word for word in tokens if not word in stopwords.words('english')]
    lemmatized_words = [ps.stem(item) for item in filtered ]
    return lemmatized_words
     