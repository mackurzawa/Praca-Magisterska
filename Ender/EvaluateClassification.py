from EnderClassifier import EnderClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from rulefit.rulefit import RuleFitClassifier


from PrepareDatasets import prepare_dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from time import time
import pandas as pd


SEED = 42
CSV_PATH = 'EVAL.csv'
KFolds = 5

ender_n_rules, ender_nu, ender_sampling = 500, 0.5, 0.5

data_final = pd.DataFrame({
    'Model': [],
    'Dataset': [],
    'Seed': [],
    'No_crossvalidation': [],
    'Time training': [],
    'Train Accuracy': [],
    'Test Accuracy': []
})
data_every_fold = data_final.copy()

datasets = [
    'haberman',
    # 'liver',
    'breast-c',
    'spambase']

for dataset in datasets:
    print()
    print('#'*100)
    print(' '*45, dataset)
    print('#'*100)
    print()
    X, y = prepare_dataset(dataset)

    X.columns = list(range(len(X.columns)))

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'XGBoost': XGBClassifier(eval_metric='logloss'),
        'CatBoost': CatBoostClassifier(verbose=0),
        'RuleFit': RuleFitClassifier(),
        'Ender_Gradient': EnderClassifier(dataset_name=dataset, n_rules=ender_n_rules, use_gradient=True, nu=ender_nu, sampling=ender_sampling, verbose=False),
        'Ender_Newton-Raphson': EnderClassifier(dataset_name=dataset, n_rules=ender_n_rules, use_gradient=False, nu=ender_nu, sampling=ender_sampling, verbose=False),
    }
    for model_name, model in models.items():
        times, train_accuracies, test_accuracies = [], [], []

        for i_crossval, (train_indices, test_indices) in enumerate(StratifiedKFold(n_splits=KFolds, shuffle=True, random_state=SEED).split(X, y)):
            X_train, y_train, X_test, y_test = X.iloc[train_indices], y[train_indices], X.iloc[test_indices], y[test_indices]

            if model_name == 'RuleFit':
                time_started = time()
                model.fit(X_train.to_numpy().astype('float'), y_train, feature_names=X.columns)
            elif model_name in ['Ender_Gradient', 'Ender_Newton-Raphson']:
                time_started = time()
                model.fit(X_train, y_train, X_test=X_test, y_test=y_test)
            else:
                time_started = time()
                model.fit(X_train, y_train)
            time_elapsed = time() - time_started
            print(f'{model_name} {i_crossval + 1} out of {KFolds}')
            print(f'Time elapsed: {time_elapsed}')

            if model_name in ['Ender_Gradient', 'Ender_Newton-Raphson']:
                model.evaluate_all_rules()
                accuracy_train = max(model.history['accuracy'])
                accuracy_test = max(model.history['accuracy_test'])
            else:
                if model_name == 'RuleFit':
                    y_pred_train = model.predict(X_train.to_numpy())
                    y_pred_test = model.predict(X_test.to_numpy())
                else:
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                accuracy_train = accuracy_score(y_train, y_pred_train)
                accuracy_test = accuracy_score(y_test, y_pred_test)

            times.append(time_elapsed)
            train_accuracies.append(accuracy_train)
            test_accuracies.append(accuracy_test)

            data_every_fold.loc[len(data_every_fold.index)] = [model_name, dataset, SEED, i_crossval + 1, time_elapsed, accuracy_train, accuracy_test]
            data_every_fold.to_csv('fold_' + CSV_PATH, index=False)
            print(f'Train accuracy: {accuracy_train}')
            print(f'Test accuracy: {accuracy_test}')
            print('='*100)
        data_final.loc[len(data_every_fold.index)] = [model_name, dataset, SEED, '-', sum(times)/len(times), sum(train_accuracies)/len(train_accuracies), sum(test_accuracies)/len(test_accuracies)]
        data_final.to_csv('final_' + CSV_PATH, index=False)
        print()
        print('=' * 100)


