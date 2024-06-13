import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def random_parameter_tune(train_val_X, train_val_y):
    n_estimators_range = [50, 100, 150, 200, 250]
    train_acc_all = []
    val_acc_all = []

    kf = KFold(n_splits = 5)
    print("Estimators | Training Accuracy  | Validation Accuracy  | std of Training  | std of Validation")
    for estimator in n_estimators_range:
        train_acc = []
        val_acc = []
        train_std_all = []
        val_std_all = []
        print("============================================================================================")
        for train_index, val_index in kf.split(train_val_X):

            train_X = train_val_X[train_index,:]
            val_X = train_val_X[val_index,:]

            train_y = train_val_y.iloc[train_index]
            val_y = train_val_y.iloc[val_index]

            rf_model = RandomForestClassifier(n_estimators=estimator)
            rf_model.fit(train_X, train_y)
            train_acc.append(rf_model.score(train_X, train_y))
            val_acc.append(rf_model.score(val_X, val_y))


        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_val_acc = sum(val_acc) / len(val_acc)
        std_train_acc = np.std(train_acc)
        std_val_acc = np.std(val_acc)
        print(f"{estimator}        | {avg_train_acc * 100}%              | { avg_val_acc * 100}%    | {std_train_acc}              | {std_val_acc}")

        train_acc_all.append(avg_train_acc)
        val_acc_all.append(avg_val_acc)
        train_std_all.append(std_train_acc)
        val_std_all.append(std_val_acc)

    return n_estimators_range, train_acc_all, val_acc_all,train_std_all,val_std_all



def random_parameter_tune_min_samples(train_val_X, train_val_y):

    min_samples_leaf_range = [1, 2, 5, 10, 20]
    train_acc_all = []
    val_acc_all = []

    kf = KFold(n_splits = 5)
    print("min samples leaf | Training Accuracy  | Validation Accuracy  | std of Training  | std of Validation")
    for min_samples_leaf in min_samples_leaf_range:
        train_acc = []
        val_acc = []
        train_std_all = []
        val_std_all = []
        print("===========================================================================================")
        for train_index, val_index in kf.split(train_val_X):

            train_X = train_val_X[train_index,:]
            val_X = train_val_X[val_index,:]

            train_y = train_val_y.iloc[train_index]
            val_y = train_val_y.iloc[val_index]

            rf_model = RandomForestClassifier(min_samples_leaf=min_samples_leaf)
            rf_model.fit(train_X, train_y)
            train_acc.append(rf_model.score(train_X, train_y))
            val_acc.append(rf_model.score(val_X, val_y))


        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_val_acc = sum(val_acc) / len(val_acc)
        std_train_acc = np.std(train_acc)
        std_val_acc = np.std(val_acc)
        print(f"       {min_samples_leaf}        | {avg_train_acc * 100}% | { avg_val_acc * 100}%    | {std_train_acc}         | {std_val_acc}")

        train_acc_all.append(avg_train_acc)
        val_acc_all.append(avg_val_acc)
        train_std_all.append(std_train_acc)
        val_std_all.append(std_val_acc)

    return min_samples_leaf_range, train_acc_all, val_acc_all, train_std_all, val_std_all