import os, sys
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

# Adjust the path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import the functions from the src module
from src import get_tfidf_vals


X_train_tfidf, X_valid_tfidf, y_train, y_valid = get_tfidf_vals()


#Updating the Decision Tree criterion parameter
criterion_values = ["gini", "entropy"]
train_accuracies = []
valid_accuracies = []

for criterion in criterion_values:
    clf = tree.DecisionTreeClassifier(criterion=criterion)
    clf.fit(X_train_tfidf, y_train)
    y_train_pred = clf.predict(X_train_tfidf)
    y_valid_pred = clf.predict(X_valid_tfidf)

    train_accuracy = accuracy_score(y_train, y_train_pred) # Calculate training and validation accuracy
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)

    train_accuracies.append(train_accuracy)
    valid_accuracies.append(valid_accuracy)

plt.figure(figsize=(8, 6))
plt.bar(criterion_values, train_accuracies, label='Training Accuracy', alpha=0.7, color='b', width=0.4)
plt.bar(criterion_values, valid_accuracies, label='Validation Accuracy', alpha=0.7, color='g', width=0.4)
plt.xlabel('Criterion')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy vs. Criterion')
plt.legend()
plt.show()

output_path = os.path.join(project_root, 'EDA.jpg')
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print("Done")
     

