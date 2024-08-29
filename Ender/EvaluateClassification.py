from EnderClassifier import EnderClassifier
from EnderClassifierFastMyImplementation import EnderClassifierFastMyImplementation
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from rulefit.rulefit import RuleFitClassifier


from PrepareDatasets import prepare_dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from time import time
import pandas as pd
import numpy as np
import multiprocessing
import os


SEED = 42
CSV_PATH = 'EVAL.csv'
KFolds = 5

# ender_n_rules = [200, 200, 50]
ender_nu, ender_sampling = 0.5, 0.25


def _mean(x):
    return sum(x) / len(x)


data_final = pd.DataFrame({
    'Model': [],
    'Dataset': [],
    'Seed': [],
    'No_crossvalidation': [],
    'Time training': [],
    'Train Accuracy': [],
    'Train Accuracy Rule Number': [],
    'Test Accuracy': [],
    'Test Accuracy Rule Number': [],
    'Train F1': [],
    'Train F1 Rule Number': [],
    'Test F1': [],
    'Test F1 Rule Number': [],
    'Number of Rules': [],
})
data_every_fold = data_final.copy()

datasets = [
    # ('apple', 200),
    # ('haberman', 200),
    # ('liver'),
    # ('breast-c', 200),
    ('spambase', 200),
]


def crossvalidation_fold(X, y, train_indices, test_indices, model, model_name, i_crossval):
    X_train, y_train, X_test, y_test = X.iloc[train_indices], y[train_indices], X.iloc[test_indices], y[test_indices]

    if model_name == 'RuleFit':
        time_started = time()
        model.fit(X_train.to_numpy().astype('float'), y_train, feature_names=X.columns)
    elif model_name in ['Ender_Gradient', 'Ender_Gradient_Fast', 'Ender_Newton-Raphson']:
        time_started = time()
        model.fit(X_train, y_train, X_test=X_test, y_test=y_test)
    else:
        time_started = time()
        model.fit(X_train, y_train)
    time_elapsed = time() - time_started
    print(f'{model_name} {i_crossval + 1} out of {KFolds}')
    print(f'Time elapsed: {time_elapsed}')

    if model_name in ['Ender_Gradient', 'Ender_Gradient_Fast', 'Ender_Newton-Raphson']:
        model.evaluate_all_rules()
        accuracy_train = max(model.history['accuracy'])
        accuracy_train_rule_number = np.argmax(model.history['accuracy'])
        accuracy_test = max(model.history['accuracy_test'])
        accuracy_test_rule_number = np.argmax(model.history['accuracy_test'])
        f1_train = max(model.history['f1'])
        f1_train_rule_number = np.argmax(model.history['f1'])
        f1_test = max(model.history['f1_test'])
        f1_test_rule_number = np.argmax(model.history['f1_test'])
        # n_rules = ender_n_rules
    else:
        if model_name == 'RuleFit':
            y_pred_train = model.predict(X_train.to_numpy())
            y_pred_test = model.predict(X_test.to_numpy())
        else:
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_train_rule_number = None
        accuracy_test = accuracy_score(y_test, y_pred_test)
        accuracy_test_rule_number = None
        f1_train = f1_score(y_train, y_pred_train)
        f1_train_rule_number = None
        f1_test = f1_score(y_test, y_pred_test)
        f1_test_rule_number = None
        # n_rules = None
    return (time_elapsed, accuracy_train, accuracy_train_rule_number, accuracy_test, accuracy_test_rule_number, f1_train, f1_train_rule_number, f1_test, f1_test_rule_number, i_crossval)


if __name__ == "__main__":
    time_start = time()
    for dataset, ender_n_rules in datasets:
        print()
        print('#'*100)
        print(' '*45, dataset)
        print('#'*100)
        print()
        X, y = prepare_dataset(dataset)

        X.columns = list(range(len(X.columns)))

        models = {
            'XGBoost': XGBClassifier(eval_metric='logloss'),
            'CatBoost': CatBoostClassifier(verbose=0),
            'RuleFit': RuleFitClassifier(random_state=SEED),
            'Ender_Gradient': EnderClassifier(dataset_name=dataset, n_rules=ender_n_rules, use_gradient=True, nu=ender_nu, sampling=ender_sampling, verbose=False),
            'Ender_Gradient_Fast': EnderClassifierFastMyImplementation(dataset_name=dataset, n_rules=ender_n_rules, use_gradient=True, nu=ender_nu, sampling=ender_sampling, verbose=False),
            'Ender_Newton-Raphson': EnderClassifier(dataset_name=dataset, n_rules=ender_n_rules, use_gradient=False, nu=ender_nu, sampling=ender_sampling, verbose=False),
        }
        for model_name, model in models.items():
            times, train_accuracies, test_accuracies, train_f1s, test_f1s = [], [], [], [], []
            args = []
            for i_crossval, (train_indices, test_indices) in enumerate(StratifiedKFold(n_splits=KFolds, shuffle=True, random_state=SEED).split(X, y)):
                args.append([X, y, train_indices, test_indices, model, model_name, i_crossval])

            with multiprocessing.Pool(processes=min(KFolds, multiprocessing.cpu_count())) as pool:
                results = pool.starmap(crossvalidation_fold, args)

            for (time_elapsed, accuracy_train, accuracy_train_rule_number, accuracy_test, accuracy_test_rule_number, f1_train, f1_train_rule_number, f1_test, f1_test_rule_number, i_crossval) in results:

                times.append(time_elapsed)
                train_accuracies.append(accuracy_train)
                test_accuracies.append(accuracy_test)
                train_f1s.append(f1_train)
                test_f1s.append(f1_test)

                data_every_fold.loc[len(data_every_fold.index)] = [model_name, dataset, SEED, i_crossval + 1, time_elapsed, accuracy_train, accuracy_train_rule_number, accuracy_test, accuracy_test_rule_number, f1_train, f1_train_rule_number, f1_test, f1_test_rule_number, ender_n_rules]
                data_every_fold.to_csv(os.path.join('..', 'fold_' + CSV_PATH), index=False)
                print(f'Train accuracy: {accuracy_train}')
                print(f'Test accuracy: {accuracy_test}')
                print(f'Train f1: {f1_train}')
                print(f'Test f1: {f1_test}')
                print('='*100)
            data_final.loc[len(data_every_fold.index)] = [model_name, dataset, SEED, '-', _mean(times), _mean(train_accuracies), None, _mean(test_accuracies), None, _mean(train_f1s), None, _mean(test_f1s), None, ender_n_rules]
            data_final.to_csv(os.path.join('..', 'final_' + CSV_PATH), index=False)
            print()
            print('=' * 100)
    print(f"Evaluated in {time() - time_start}")


