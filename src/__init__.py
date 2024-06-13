import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from src.preprocessing import get_processed_tokens,get_processed_words_gb

def get_train_data():
    """
    Load the train dataset.
    
    Returns:
        pd.DataFrame: The train dataset.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_csv_path = os.path.join(project_root, 'data', 'news-train-1.csv')
    train_df = pd.read_csv(train_csv_path)
    return train_df

def get_test_data():
    """
    Load the test dataset.
    
    Returns:
        pd.DataFrame: The test dataset.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    test_csv_path = os.path.join(project_root, 'data', 'news-test.csv')
    test_df = pd.read_csv(test_csv_path)
    return test_df

def get_preprocessed_train_data():
    """
    Load and preprocess the train dataset.
    
    Returns:
        pd.DataFrame: The preprocessed train dataset.
    """
    train_df = get_train_data()
    train_df['cleaned_text'] = train_df['Text'].apply(lambda x: get_processed_tokens(x))
    return train_df

def get_preprocessed_test_data():
    """
    Load and preprocess the test dataset.
    
    Returns:
        pd.DataFrame: The preprocessed test dataset.
    """
    test_df = get_test_data()
    test_df['cleaned_text'] = test_df['Text'].apply(lambda x: get_processed_tokens(x))
    return test_df


def get_preprocessed_train_data_gb():
    """
    Load and preprocess the train dataset.
    
    Returns:
        pd.DataFrame: The preprocessed train dataset.
    """
    train_df = get_train_data()
    train_df['cleaned_text'] = train_df['Text'].apply(lambda x: get_processed_words_gb(x))
    return train_df

def get_preprocessed_test_data_gb():
    """
    Load and preprocess the test dataset.
    
    Returns:
        pd.DataFrame: The preprocessed test dataset.
    """
    test_df = get_test_data()
    test_df['cleaned_text'] = test_df['Text'].apply(lambda x: get_processed_words_gb(x))
    return test_df


def get_tfidf_vals():
    
    train_df = get_preprocessed_train_data()
    X = train_df['cleaned_text']  # Features (cleaned_text)
    y = train_df['Category']  # Target variable

    # Randomly Splitting the data into training and testing ---> 80% train, 20% test
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_valid_tfidf = tfidf_vectorizer.transform(X_valid)

    return X_train_tfidf,X_valid_tfidf,y_train,y_valid


def get_countVect_vals():
    
    test_data = get_preprocessed_test_data()
    X_test = test_data['cleaned_text']

    train_df = get_preprocessed_train_data()
    X_train = train_df['cleaned_text']  # Features (cleaned_text)
    y_train = train_df['Category']  # Target variable

    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    return X_train_vectorized,X_test_vectorized,test_data





