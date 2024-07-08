import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PrepareDatasets import prepare_dataset
from time import time

#######################
# dataset = 'haberman'
# dataset = 'liver'
# dataset = 'breast-c'
dataset = 'spambase'

X, y = prepare_dataset(dataset)

X.columns = list(range(len(X.columns)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

time_started = time()
model.fit(X_train, y_train)
time_elapsed = time() - time_started

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f'Train accuracy: {accuracy_train}')
print(f'Test accuracy: {accuracy_test}')
print(f'Time elapsed: {time_elapsed}')
