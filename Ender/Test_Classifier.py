from EnderClassifier import EnderClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from PrepareDatasets import prepare_wine_classification_dataset, prepare_apple_classification_dataset, prepare_apple_small_classification_dataset


n_rules = 100
use_gradient = True


def calculate_accuracy_from_probabilities(y, y_preds):
    y_pred_labels = [np.argmax(y_pred) for y_pred in y_preds]
    print("Accuracy:")
    print(accuracy_score(y, y_pred_labels))


X, y = prepare_wine_classification_dataset()
# X, y = prepare_apple_small_classification_dataset()
# X, y = prepare_apple_classification_dataset()

ender = EnderClassifier(n_rules=n_rules, use_gradient=use_gradient)
ender.fit(X, y)

y_preds = ender.predict(X)
calculate_accuracy_from_probabilities(y, y_preds)
