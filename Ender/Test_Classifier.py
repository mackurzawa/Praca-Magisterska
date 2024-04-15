from EnderClassifier import EnderClassifier
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score


DATA_PATH = os.path.join('..', 'data')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'test_data')

# squared error loss działa perfett, absolute error po kilku lekko odbiega
n_rules = 100
loss = 'squared_error_loss_function'  # -> in Weka called lossFunction = Squared Error Loss
# loss = 'absolute_error_loss_function' # almost working, after few it changes a bit, in weka called Absolute Error Loss
empirical_risk_minimizer = 'gradient_empirical_risk_minimizer'  # -. in Weka called minimizationTechnique = Simultaneus minimization
# empirical_risk_minimizer = 'absolute_error_risk_minimizer'  # -> in weka called minimizationTechnique = Gradient Ddscent


def calculate_accuracy_from_probabilities(y, y_preds):
    y_pred_labels = [np.argmax(y_pred) for y_pred in y_preds]
    print("Accuracy:")
    print(accuracy_score(y, y_pred_labels))


def map_classes(series, class_mapping):
    mapped_series = series.map(class_mapping)
    return mapped_series


def prepare_wine_classification_dataset():
    data = pd.read_csv(os.path.join(DATA_PATH, "Classification Wine - 3 cl. 13 cols. 178 ex.csv"))

    decision_attribute = "Wine"
    X, y = data.drop([decision_attribute], axis=1), data[decision_attribute]
    X = pd.get_dummies(X)
    class_mapping = {'Class1': 0, 'Class2': 1, 'Class3': 2}
    y = np.array(map_classes(y, class_mapping))
    return X, y


def prepare_apple_small_classification_dataset():
    data = pd.read_csv(os.path.join(DATA_PATH, "Classification Apple - 2 cl. 8 cols. 100 ex.csv"))

    decision_attribute = "Quality"
    X, y = data.drop([decision_attribute], axis=1), data[decision_attribute]
    X = pd.get_dummies(X)
    class_mapping = {'bad': 0, 'good': 1}
    y = np.array(map_classes(y, class_mapping))
    return X, y


def prepare_apple_classification_dataset():
    data = pd.read_csv(os.path.join(DATA_PATH, "Classification Apple - 2 cl. 8 cols. 4000 ex.csv"))

    decision_attribute = "Quality"
    X, y = data.drop([decision_attribute], axis=1), data[decision_attribute]
    X = pd.get_dummies(X)
    class_mapping = {'bad': 0, 'good': 1}
    y = np.array(map_classes(y, class_mapping))
    return X, y



X, y = prepare_wine_classification_dataset()
# X, y = prepare_apple_small_classification_dataset()
# X, y = prepare_apple_classification_dataset()

ender = EnderClassifier(n_rules=n_rules, loss=loss, empirical_risk_minimizer=empirical_risk_minimizer)
ender.fit(X, y)

y_preds = ender.predict(X)
calculate_accuracy_from_probabilities(y, y_preds)
