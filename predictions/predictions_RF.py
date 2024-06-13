import os,sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


# Adjust the path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import the functions from the src module
from model.randomforest import random_parameter_tune, random_parameter_tune_min_samples
from src import get_tfidf_vals, get_countVect_vals

X_train_tfidf, X_valid_tfidf, y_train, y_valid = get_tfidf_vals()

n_estimators_range, train_acc_all, val_acc_all, train_std_all, val_std_all = random_parameter_tune(X_train_tfidf, y_train)

# plot training/validation curves
print("\n")
plt.plot(n_estimators_range, train_acc_all, marker='.', label="Training accuracy")
plt.plot(n_estimators_range, val_acc_all, marker='.', label="Validation accuracy")
plt.xlabel('Estimator values')
plt.ylabel('Accuracy')
plt.legend()


min_samples_leaf, train_acc_all, val_acc_al, train_std_all, val_std_alll = random_parameter_tune_min_samples(X_train_tfidf, y_train)

# plot training/validation curves
print("\n")
plt.plot(min_samples_leaf, train_acc_all, marker='.', label="Training accuracy")
plt.plot(min_samples_leaf, val_acc_all, marker='.', label="Validation accuracy")
plt.xlabel('Minimum Sample Leaves')
plt.ylabel('Accuracy')
plt.legend()


# Create a CountVectorizer to convert text data into numerical features
X_train_vectorized, X_test_vectorized, test_data = get_countVect_vals()

rf_classifier = RandomForestClassifier(n_estimators=250)
rf_classifier.fit(X_train_vectorized, y_train)
y_pred = rf_classifier.predict(X_test_vectorized)
test_data['predicted_label'] = y_pred

final_data = test_data[['ArticleId','predicted_label']]

#saving the tested data with predicted labels to a CSV file
final_data.to_csv('labels1.csv', index=False,header=False)