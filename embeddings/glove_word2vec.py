import numpy as np
import os,sys
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.preprocessing import LabelEncoder

#you can find different versions of Glove in below URL:
# http://nlp.stanford.edu/data/glove.6B.zip Dowload Glove file: https://nlp.stanford.edu/projects/glove/


glove_input_file = 'glove.6B.50d.txt'
word2vec_output_file = 'glove.6B.50d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)
glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)


# Adjust the path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import the functions from the src module
from predictions import predictions_NN as pred
from src import get_preprocessed_train_data

train_df = get_preprocessed_train_data()
X = train_df['cleaned_text']
y = train_df['Category']
X_train = train_df['cleaned_text'].copy()
y_train = train_df['Category'].copy()

# Create word embeddings for text data
X_vectors = []
for text in X:
    # Split the text into words
    words = text.split()
    vectors = [glove_model[word] if word in glove_model else np.zeros(50) for word in words]
    if vectors:
        avg_vector = np.mean(vectors, axis=0)
    else:
        avg_vector = np.zeros(50)  # GloVe vectors are 50-dimensional
    X_vectors.append(avg_vector)
X_vectors = np.array(X_vectors)


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
pred.perform_k_fold_CV(X_vectors, y_train_encoded,"glove2word2vec")