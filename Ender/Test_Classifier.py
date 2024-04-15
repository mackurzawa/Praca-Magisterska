from EnderClassifier import EnderClassifier
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score


DATA_PATH = os.path.join('..', 'data')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'test_data')

# squared error loss działa perfett, absolute error po kilku lekko odbiega
n_rules = 10
loss = 'squared_error_loss_function'  # -> in Weka called lossFunction = Squared Error Loss
# loss = 'absolute_error_loss_function' # almost working, after few it changes a bit, in weka called Absolute Error Loss
empirical_risk_minimizer = 'gradient_empirical_risk_minimizer'  # -. in Weka called minimizationTechnique = Simultaneus minimization
# empirical_risk_minimizer = 'absolute_error_risk_minimizer'  # -> in weka called minimizationTechnique = Gradient Ddscent


def map_classes(series):
    # Tworzymy słownik mapujący wartości klas na odpowiednie liczby
    class_mapping = {'Class1': 0, 'Class2': 1, 'Class3': 2}
    # Używamy metody map w celu zamiany wartości na odpowiadające im liczby zgodnie ze słownikiem
    mapped_series = series.map(class_mapping)
    return mapped_series


def start_small_classification():
    data = pd.read_csv(os.path.join(TEST_DATA_PATH, "small classification dataset.csv"))

    decision_attribute = "DecyzyjnyAtrybut"
    X, y = data.drop([decision_attribute], axis=1), np.array(data[decision_attribute].astype(int))

    ender = EnderClassifier(n_rules=n_rules, loss=loss, empirical_risk_minimizer=empirical_risk_minimizer)
    ender.fit(X, y)
    # print(ender.predict([[1, 3, 2]]))


def calculate_accuracy_from_probabilities(y, y_preds):
    y_pred_labels = [np.argmax(y_pred) for y_pred in y_preds]
    print("Accuracy:")
    print(accuracy_score(y, y_pred_labels))


def start_wine_classification():
    data = pd.read_csv(os.path.join(DATA_PATH, "Classification Wine - 3.csv"))

    decision_attribute = "Wine"
    X, y = data.drop([decision_attribute], axis=1), data[decision_attribute]
    X = pd.get_dummies(X)
    y = np.array(map_classes(y))
    # print(y)

    ender = EnderClassifier(n_rules=n_rules, loss=loss, empirical_risk_minimizer=empirical_risk_minimizer)
    ender.fit(X, y)

    y_preds = ender.predict(X)
    calculate_accuracy_from_probabilities(y, y_preds)


# start_small_classification()
start_wine_classification()
