import sys
sys.path.append('../Ender')
from rulefit.rulefit import RuleFitClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Ender.PrepareDatasets import prepare_dataset
from time import time

#######################
# dataset = 'haberman'
# dataset = 'liver'
# dataset = 'breast-c'
dataset = 'spambase'

X, y = prepare_dataset(dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

rf = RuleFitClassifier()

time_started = time()
rf.fit(X_train, y_train, feature_names=X.columns)
time_elapsed = time() - time_started
print(f'Time elapsed: {time_elapsed}')

y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f'Train accuracy: {accuracy_train}')
print(f'Test accuracy: {accuracy_test}')
