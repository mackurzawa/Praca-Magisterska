import pickle
from time import time
import os
import random

import mlflow
from sklearn.model_selection import train_test_split


from EnderClassifier import EnderClassifier
from PrepareDatasets import prepare_dataset
from CalculateMetrics import calculate_all_metrics
from VisualiseHistory import visualise_history
from multiprocessing import Pool


if __name__ == "__main__":
    RANDOM_STATE = 42
    n_rules = 100
    use_gradient = True
    # use_gradient = False
    optimized_searching_for_cut = 0  # Standard
    optimized_searching_for_cut = 1  # Quicker
    # optimized_searching_for_cut = 2  # The quickest
    prune = False
    # TRAIN_NEW = False
    TRAIN_NEW = True
    dataset = 'apple'  # to samo dla 3 searching przy 500 regułach
    # dataset = 'wine'  # inaczej
    ##########
    # dataset = 'haberman' #inaczej
    # dataset = 'liver'
    # dataset = 'breast-c' # inaczej
    # dataset = 'spambase' # inaczej

    nu = .5
    sampling = .5
    # sampling = 1

    params = {
        "Classification": True,
        "Dataset": dataset,
        "n_rules": n_rules,
        "Gradient. not Newton_Raphson": use_gradient,
        "Shrinkage. nu": nu,
        "Sampling. percentage": sampling,
    }
    random.seed(42)
    with mlflow.start_run() as run:
        mlflow.log_params(params)

        X, y = prepare_dataset(dataset)
        # print(X)
        # print(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # X_train = X
        # X_test = []
        # y_train = y
        # y_test = []
        if TRAIN_NEW:
            ender = EnderClassifier(dataset_name=dataset, n_rules=n_rules, use_gradient=use_gradient, optimized_searching_for_cut=optimized_searching_for_cut, nu=nu, sampling=sampling, random_state=RANDOM_STATE)
            # ender.pool = Pool()

            FILENAME = f'log{optimized_searching_for_cut}.txt'
            # file = open(FILENAME, 'w')
            # ender.file = file
            time_started = time()
            ender.fit(X_train, y_train, X_test=X_test, y_test=y_test)
            # ender.fit(X_train, y_train)
            time_elapsed = round(time() - time_started, 2)
            # file.close()
            mlflow.log_metric("Training time", time_elapsed)
            print(f"Rules created in {time_elapsed} s.")
            time_started = time()
            ender.evaluate_all_rules()
            time_elapsed = round(time() - time_started, 2)
            mlflow.log_metric("Evaluation time", time_elapsed)
            print(f"Rules evaluated in {time_elapsed} s.")

            with open(os.path.join('models', f'model_{dataset}_{n_rules}.pkl'), 'wb') as f:
                pickle.dump(ender, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join('models', f'model_{dataset}_{n_rules}.pkl'), 'rb') as f:
                ender = pickle.load(f)

        rules_order = [3, 46, 26, 61, 23, 48, 2, 17, 60, 31, 21, 85, 38, 32, 77, 96, 4, 28, 70, 69, 15, 57, 41, 90, 62, 39, 51, 14, 64, 9, 30, 66, 50, 1, 0, 29, 11, 7, 54, 47, 99, 36, 92, 5, 84, 6, 72, 98, 95, 55, 73, 44, 24, 67, 89, 42, 91, 74, 86, 93, 25, 18, 68, 27, 76, 16, 87, 82, 83, 12, 52, 33, 97, 35, 43, 59, 71, 20, 8, 45, 53, 94, 40, 19, 81, 10, 22, 80, 78, 56, 34, 37, 65, 58, 49, 13, 75, 88, 79, 63]
        # rules_order = [21, 79, 48, 12, 86, 90, 61, 60, 52, 95, 72, 8, 97, 73, 84, 62, 7, 91, 31, 80, 77, 74, 66, 39, 67, 87, 76, 22, 23, 81, 83, 13, 94, 25, 55, 54, 82, 89, 70, 99, 93, 36, 88, 46, 0, 32, 30, 69, 2, 98, 6, 92, 11, 63, 37, 57, 68, 56, 53, 85, 75, 1, 65, 18, 14, 78, 33, 51, 58, 44, 38, 49, 26, 40, 29, 34, 45, 64, 17, 42, 10, 28, 27, 4, 43, 71, 59, 47, 41, 35, 24, 20, 19, 16, 15, 9, 5, 50, 96]
        for i in range(len(rules_order)+1):
            print(calculate_all_metrics(y_test, ender.predict_with_specific_rules(X_test, rules_order[:i+1])))

        y_train_preds = ender.predict(X_train, use_effective_rules=False)
        final_metrics_train = calculate_all_metrics(y_train, y_train_preds)
        print("Before pruning train:", final_metrics_train)
        y_test_preds = ender.predict(X_test, use_effective_rules=False)
        final_metrics_test = calculate_all_metrics(y_test, y_test_preds)
        print("Before pruning test: ", final_metrics_test)