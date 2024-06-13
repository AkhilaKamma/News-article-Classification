import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn import tree


def dtc_parameter_tune(train_val_X, train_val_y):
    min_samples_leaf = [1, 3, 5, 7, 10, 50, 100, 200]
    train_acc_all = []
    val_acc_all = []
    train_std_all = []
    val_std_all = []

    kf = KFold(n_splits = 5)
    print("min_samples_leaf | Training Accuracy  | Validation Accuracy  | std of Training  | std of Validation")
    for sample in min_samples_leaf:
        train_acc = []
        val_acc = []
        print("================================================================================================")
        for train_index, val_index in kf.split(train_val_X):

            train_X = train_val_X[train_index,:]
            val_X = train_val_X[val_index,:]

            train_y = train_val_y.iloc[train_index]
            val_y = train_val_y.iloc[val_index]

            dtc = tree.DecisionTreeClassifier(min_samples_leaf=sample)
            dtc.fit(train_X, train_y)
            train_acc.append(dtc.score(train_X, train_y))
            val_acc.append(dtc.score(val_X, val_y))


        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_val_acc = sum(val_acc) / len(val_acc)
        std_train_acc = np.std(train_acc)
        std_val_acc = np.std(val_acc)
        print(f"{sample} | {avg_train_acc * 100}% | { avg_val_acc * 100}%| {std_train_acc}| {std_val_acc}")

        train_acc_all.append(avg_train_acc)
        val_acc_all.append(avg_val_acc)
        train_std_all.append(std_train_acc)
        val_std_all.append(std_val_acc)

    return min_samples_leaf, train_acc_all, val_acc_all,train_std_all,val_std_all



def dtc_parameter_tune_features(train_val_X, train_val_y):
    max_features_values = [None, 'sqrt', 'log2']
    train_acc_all = []
    val_acc_all = []

    kf = KFold(n_splits = 5)
    print("max_features | Training Accuracy  | Validation Accuracy  | std of Training  | std of Validation")
    for max_feature in max_features_values:
        train_acc = []
        val_acc = []
        train_std_all = []
        val_std_all = []
        print("=============================================================================================")
        for train_index, val_index in kf.split(train_val_X):

            train_X = train_val_X[train_index,:]
            val_X = train_val_X[val_index,:]

            train_y = train_val_y.iloc[train_index]
            val_y = train_val_y.iloc[val_index]

            dtc = tree.DecisionTreeClassifier(max_features=max_feature)
            dtc.fit(train_X, train_y)
            train_acc.append(dtc.score(train_X, train_y))
            val_acc.append(dtc.score(val_X, val_y))


        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_val_acc = sum(val_acc) / len(val_acc)
        std_train_acc = np.std(train_acc)
        std_val_acc = np.std(val_acc)
        print(f"{max_feature}        | {avg_train_acc * 100}%              | { avg_val_acc * 100}%    | {std_train_acc}            | {std_val_acc}")

        train_acc_all.append(avg_train_acc)
        val_acc_all.append(avg_val_acc)
        train_std_all.append(std_train_acc)
        val_std_all.append(std_val_acc)

    return max_features_values, train_acc_all, val_acc_all,train_std_all,val_std_all