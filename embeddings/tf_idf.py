import pandas as pd
import os
import numpy as np
import sys


# Adjust the path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import the functions from the src module
from src import get_preprocessed_train_data, get_preprocessed_test_data

# Load the preprocessed datasets
train_df = get_preprocessed_train_data()
test_df = get_preprocessed_test_data()


final_corpus = train_df['cleaned_text'].to_list()

term_freq_matrix = []
lexicons = set()
for word_list in final_corpus:
    lexicons.update(word_list)
lexicons = sorted(list(lexicons))
word_to_index = {word: index for index, word in enumerate(lexicons)}


##Creating the Tf matrix while maintaining document order

for word_list in final_corpus:
    vect = [0] * len(lexicons)
    for word in word_list:
        if word in word_to_index:
            vect[word_to_index[word]] += 1
    max_tf = max(vect)
    tf_vector_normalized = [tf / max_tf if max_tf > 0 else 0 for tf in vect]
    term_freq_matrix.append(tf_vector_normalized)

#creating IDF vector
num_documents = len(final_corpus)
df = [0] * len(lexicons)
for words_list in final_corpus:
    set_of_words = set(words_list)
    for word in set_of_words:
        if word in word_to_index:
            df[word_to_index[word]] += 1
idf_vector = [0 if df_value == 0 else np.log(num_documents / df_value) for df_value in df]


# Calculate TF-IDF matrix
tfidf_matrix = []
for vect in term_freq_matrix:
    tfidf_vector = [tf * idf for tf, idf in zip(vect, idf_vector)]
    tfidf_matrix.append(tfidf_vector)

# # Convert the TF-IDF matrix to a pandas DataFrame
columns = lexicons
tfidf_df = pd.DataFrame(tfidf_matrix,columns=columns)



output_path = os.path.join(project_root, 'output', 'matrix.txt')
# Save the TF-IDF matrix to a text file without column names
with open(output_path, 'w', encoding='utf-8') as f:
    for row in tfidf_matrix:
        f.write(','.join(map(str, row)) + '\n')

print("Output saved to matrix.txt")