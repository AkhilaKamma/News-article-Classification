from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec




# Create a doc2Vectorizer to convert text data into numerical features ----> Used ChatGpt to understand this Concept
tagged_data_train = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(X_train)]
tagged_data_valid = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(X_test)]

# Initialize and train the Doc2Vec model
doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=20)
doc2vec_model.build_vocab(tagged_data_train)
doc2vec_model.train(tagged_data_train, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

X_train_doc2vec = [doc2vec_model.infer_vector(doc.words) for doc in tagged_data_train]
X_valid_doc2vec = [doc2vec_model.infer_vector(doc.words) for doc in tagged_data_valid]


gb_classifier = GradientBoostingClassifier(n_estimators=500, min_samples_split=10, max_features=1000, random_state=42)
gb_classifier.fit(X_train_doc2vec, y_train)
Y_train_pred = gb_classifier.predict(X_train_doc2vec)

train_accuracy = accuracy_score(y_train, Y_train_pred)
print(f"Training Accuracy: {train_accuracy:.2f}")

Y_valid_pred = gb_classifier.predict(X_valid_doc2vec)
test_data['predicted_label'] = Y_valid_pred
final_data = test_data[['ArticleId','predicted_label']]

final_data.to_csv('labels.csv', index=False,header=False)
     