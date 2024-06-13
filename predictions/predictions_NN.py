
import torch,sys,os
from torch import nn
import pandas as pd
from torch import optim
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader,TensorDataset



# Adjust the path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import the functions from the src module
from model import neuralnetwork as nnet


features_list = []
train_accuracy_scores = []
test_accuracy_scores = []
train_accuracy_stddev = []
test_accuracy_stddev = []
def perform_k_fold_CV(X_train, y_train,feature_method):
    # Split the data using 5-fold cross-validation
    kf = KFold(n_splits=5)
    train_per = []
    val_per = []

    for ind1, ind2 in kf.split(np.array(X_train)):
        #initialize the model for each fold
        model = nnet.NeuralNetwork(X_train.shape[1], 128, 128, len(np.unique(y_train)))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        X_train_split, X_val_split = np.array(X_train)[ind1], np.array(X_train)[ind2]
        y_train_split, y_val_split = np.array(y_train)[ind1], np.array(y_train)[ind2]

        # Convert data to DataLoader format
        train_loader,val_loader = nnet.data_format(X_train_split,y_train_split,X_val_split,y_val_split)

        # Train the model
        nnet.train_model(model,optimizer,train_loader,loss_fn)

        # Evaluate on training and validation data
        train_per.append(nnet.evaluate(model, train_loader))
        val_per.append(nnet.evaluate(model, val_loader))

    print(f"Train accuraccy: {train_per}")
    print(f"Evaluation accuraccy: {val_per}")

    #store all the accuracies
    features_list.append(feature_method)
    train_accuracy_scores.append(np.mean(train_per))
    test_accuracy_scores.append(np.mean(val_per))
    train_accuracy_stddev.append(np.std(train_per))
    test_accuracy_stddev.append(np.std(val_per))
     

def train_model_with_optimizers(X_train, y_train, optimizers):
    avg_train_accuracies = []
    std_train_accuracies = []
    avg_val_accuracies = []
    std_val_accuracies = []

    for optimizer_name in optimizers:
        # Split the data using 5-fold cross-validation
        kf = KFold(n_splits=5)
        train_accuracies = []
        val_accuracies = []
        leaning_rate = 0.001

        for ind1, ind2 in kf.split(np.array(X_train)):
            #initialize the model for each fold
            model = nnet.NeuralNetwork(X_train.shape[1], 128, 128, len(np.unique(y_train)))
            loss_fn = nn.CrossEntropyLoss()
            optimizer_map = {
                    "SGD": optim.SGD,
                    "Adam": optim.Adam,
                    "RMSprop": optim.RMSprop
            }

            if optimizer_name in optimizer_map:
                optimizer = optimizer_map[optimizer_name](model.parameters(), lr=leaning_rate)
            else:
                print(f"Not a Valid optimizer: {optimizer_name}")

            X_train_split, X_val_split = np.array(X_train)[ind1], np.array(X_train)[ind2]
            y_train_split, y_val_split = np.array(y_train)[ind1], np.array(y_train)[ind2]

            # Convert data to DataLoader format
            train_loader,val_loader = nnet.data_format(X_train_split,y_train_split,X_val_split,y_val_split)

            # Train the model
            nnet.train_model(model,optimizer,train_loader,loss_fn)

            # Evaluate on training and validation data
            train_accuracies.append(nnet.evaluate(model, train_loader))
            val_accuracies.append(nnet.evaluate(model, val_loader))

        avg_train_accuracies.append(np.mean(train_accuracies))
        std_train_accuracies.append(np.std(train_accuracies))
        avg_val_accuracies.append(np.mean(val_accuracies))
        std_val_accuracies.append(np.std(val_accuracies))

    results_dataframe = pd.DataFrame({
        "Optimizers": optimizers,
        "Avg Train Accuracy": avg_train_accuracies,
        "Std Train Accuracy": std_train_accuracies,
        "Avg Val Accuracy": avg_val_accuracies,
        "Std Val Accuracy": std_val_accuracies,
    })
    sns.set_style("whitegrid")
    #display(results_dataframe)
    return avg_train_accuracies,avg_val_accuracies
     