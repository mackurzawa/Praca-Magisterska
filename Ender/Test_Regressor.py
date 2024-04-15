from EnderClassifier import EnderClassifier
from EnderRegressor import EnderRegressor
import pandas as pd
import numpy as np
import os


DATA_PATH = os.path.join('..', 'data')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'test_data')

# squared error loss działa perfett, absolute error po kilku lekko odbiega
n_rules = 25
loss = 'squared_error_loss_function'  # -> in Weka called lossFunction = Squared Error Loss
# loss = 'absolute_error_loss_function' # almost working, after few it changes a bit, in weka called Absolute Error Loss
empirical_risk_minimizer = 'gradient_empirical_risk_minimizer'  # -. in Weka called minimizationTechnique = Simultaneus minimization
# empirical_risk_minimizer = 'absolute_error_risk_minimizer'  # -> in weka called minimizationTechnique = Gradient Ddscent


def start_small_regression():
    data = pd.read_csv(os.path.join(TEST_DATA_PATH, "small regression dataset.csv"))

    decision_attribute = "DecyzyjnyAtrybut"
    X, y = data.drop([decision_attribute], axis=1), np.array(data[decision_attribute])

    ender = EnderRegressor(n_rules=n_rules, loss=loss, empirical_risk_minimizer=empirical_risk_minimizer)
    ender.fit(X, y)
    # print(ender.predict([[1, 3, 2]]))


def start_housing_regression():
    data = pd.read_csv(os.path.join(DATA_PATH, "housingWithoutMissingValues.csv"))

    decision_attribute = "median_house_value"
    data = pd.get_dummies(data)
    X, y = data.drop([decision_attribute], axis=1), np.array(data[decision_attribute])

    ender = EnderRegressor(n_rules=n_rules, loss=loss, empirical_risk_minimizer=empirical_risk_minimizer)
    ender.fit(X, y)

start_housing_regression()
# start_small_regression()
