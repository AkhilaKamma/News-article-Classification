import torch,sys,os
from torch import nn
import pandas as pd
from torch import optim
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader,TensorDataset



# Adjust the path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import the functions from the src module
from model import neuralnetwork as nnet
from src import get_preprocessed_train_data,get_preprocessed_test_data

train_df = get_preprocessed_train_data()
test_df = get_preprocessed_test_data()


def test_model(X_train, y_train, X_test):

    # Initialize the model
    model = nnet.NeuralNetwork(X_train.shape[1], 128, 128, len(np.unique(y_train)))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert training data to DataLoader format
    train_data = nnet.Data(X_train, y_train)
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

    # Convert test data to DataLoader format
    test_data = nnet.Data(X_test, np.zeros(X_test.shape[0]))
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    ## Train the model
    nnet.train_model(model,optimizer,train_loader,loss_fn)

    acc = nnet.evaluate(model, train_loader)

    print(f"Train accuraccy: {acc}")

    # Make predictions on the test data
    predictions = []
    with torch.no_grad():
        for X, _ in test_loader:
            pred = model(X)
            predictions.extend(torch.argmax(pred, dim=1).tolist())

    return predictions


X_train = train_df['cleaned_text'].copy()
y_train = train_df['Category'].copy()
X_test = test_df['cleaned_text'].copy()

test_df = test_df[["ArticleId","Category"]]

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
X_train_tfidf = X_train_tfidf.toarray()
X_test_tfidf = X_test_tfidf.toarray()


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Convert data to numpy arrays
X_train_tfidf = np.array(X_train_tfidf)
y_train_encoded = np.array(y_train_encoded)
X_test_tfidf = np.array(X_test_tfidf)
predictions = test_model(X_train_tfidf, y_train_encoded,X_test_tfidf)


# Decode the predictions back to their original category labels
y_test_predicted = label_encoder.inverse_transform(predictions)
test_df['Category'] = y_test_predicted


test_df.to_csv('output/labels.csv', index=False,header=False)
     
     