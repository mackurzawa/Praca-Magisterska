from EnderRegressor import EnderRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import os
from sklearn.utils.validation import check_X_y


DATA_PATH = os.path.join('..', 'data')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'test_data')

n_rules = 100
loss = 'squared_error_loss_function'
# loss = 'absolute_error_loss_function'
empirical_risk_minimizer = 'gradient_empirical_risk_minimizer'
# empirical_risk_minimizer = 'absolute_error_risk_minimizer'


data = pd.read_csv(os.path.join(DATA_PATH, "Regression Insurance.csv"))
data = pd.read_csv(os.path.join(DATA_PATH, "Regression LiverDisorders.csv"))
decision_attribute = "drinks"
data = pd.get_dummies(data)
X, y = data.drop([decision_attribute], axis=1), np.array(data[decision_attribute])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
for x in X.iterrows():
    print(x, type(x))

ender = EnderRegressor(n_rules=n_rules, loss=loss, empirical_risk_minimizer=empirical_risk_minimizer)
ender.fit(X_train, y_train)
ender.dataset_name = "Medical Cost"
ender.dataset_name = "Liver Disorders"

preds = ender.predict(X_test)
print(mean_absolute_error(y_test, preds))
print(mean_squared_error(y_test, preds))
print((mean_squared_error(y_test, preds))**(1/2))

X_train, y_train = check_X_y(X_train, y_train)
X_test, y_test = check_X_y(X_test, y_test)
ender.prune_rules(X_train, X_test, y_train, y_test)

