import os,sys
import matplotlib.pyplot as plt

# Adjust the path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import the functions from the src module
from model.decisiontree import dtc_parameter_tune,dtc_parameter_tune_features
from src import get_tfidf_vals


X_train_tfidf, X_valid_tfidf, y_train, y_valid = get_tfidf_vals()



min_samples_leaf, train_acc_all, val_acc_all, train_std_all, val_std_all = dtc_parameter_tune(X_train_tfidf, y_train)

#Plotting the training and validation accuracies w.r.t min samples leaf
print("\n")
plt.plot(min_samples_leaf, train_acc_all, marker='.', label="Training accuracy")
plt.plot(min_samples_leaf, val_acc_all, marker='.', label="Validation accuracy")
plt.xlabel('Min samples leaf of tree')
plt.ylabel('Accuracy')
plt.legend()


max_features_values, train_acc_all, val_acc_all,train_std_all,val_std_all = dtc_parameter_tune_features(X_train_tfidf, y_train)

# plot training/validation curves
print("\n")
max_features_values = ['NO features', 'sqrt', 'log2']
plt.plot(max_features_values, train_acc_all, marker='.', label="Training accuracy")
plt.plot(max_features_values, val_acc_all, marker='.', label="Validation accuracy")
plt.xlabel('Max feature values')
plt.ylabel('Accuracy')
plt.legend()


