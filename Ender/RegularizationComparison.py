import pickle
from time import time
import os
import random
import matplotlib.pyplot as plt

import mlflow
import numpy as np
from sklearn.model_selection import train_test_split


from EnderClassifier import EnderClassifier
from PrepareDatasets import prepare_dataset
from CalculateMetrics import calculate_all_metrics
# from textwrap import wrap


if __name__ == "__main__":
    RANDOM_STATE = 42
    n_rules = 200
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

    nus = [1, .5, .25]
    samplings = [1, .5, .25]

    random.seed(42)

    accuracies_train = np.zeros((len(nus), len(samplings)))
    accuracies_train_all = np.zeros((len(nus), len(samplings), n_rules+1))
    accuracies_test = np.zeros((len(nus), len(samplings)))
    accuracies_test_all = np.zeros((len(nus), len(samplings), n_rules+1))
    times = np.zeros((len(nus), len(samplings)))
    for nu in nus:
        for sampling in samplings:

            X, y = prepare_dataset(dataset)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

            if TRAIN_NEW:
                ender = EnderClassifier(dataset_name=dataset, n_rules=n_rules, use_gradient=use_gradient, optimized_searching_for_cut=optimized_searching_for_cut, nu=nu, sampling=sampling, random_state=RANDOM_STATE)

                time_started = time()
                ender.fit(X_train, y_train, X_test=X_test, y_test=y_test)
                time_elapsed = round(time() - time_started, 2)
                mlflow.log_metric("Training time", time_elapsed)
                # print(f"Rules created in {time_elapsed} s.")
                # time_started = time()
                ender.evaluate_all_rules()
                # time_elapsed = round(time() - time_started, 2)
                mlflow.log_metric("Evaluation time", time_elapsed)
                # print(f"Rules evaluated in {time_elapsed} s.")

                with open(os.path.join('models', f'model_{dataset}_{n_rules}.pkl'), 'wb') as f:
                    pickle.dump(ender, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(os.path.join('models', f'model_{dataset}_{n_rules}.pkl'), 'rb') as f:
                    ender = pickle.load(f)

            y_train_preds = ender.predict(X_train, use_effective_rules=False)
            final_metrics_train = calculate_all_metrics(y_train, y_train_preds)
            # print("Before pruning train:", final_metrics_train)
            y_test_preds = ender.predict(X_test, use_effective_rules=False)
            final_metrics_test = calculate_all_metrics(y_test, y_test_preds)
            # print("Before pruning test: ", final_metrics_test)
            mlflow.log_metric("Last Accuracy Train", final_metrics_train['accuracy'])
            mlflow.log_metric("Last Accuracy Test", final_metrics_test['accuracy'])
            mlflow.log_metric("Max Accuracy Train", max(ender.history['accuracy']))
            mlflow.log_metric("Max Accuracy Test", max(ender.history['accuracy_test']))
            accuracies_train[nus.index(nu)][samplings.index(sampling)] = final_metrics_train['accuracy']
            accuracies_train_all[nus.index(nu)][samplings.index(sampling)] = ender.history['accuracy']
            accuracies_test[nus.index(nu)][samplings.index(sampling)] = final_metrics_test['accuracy']
            accuracies_test_all[nus.index(nu)][samplings.index(sampling)] = ender.history['accuracy_test']
            times[nus.index(nu)][samplings.index(sampling)] = time_elapsed
    print(accuracies_train)
    print('='*100)
    print(accuracies_test)
    print('='*100)
    print(times)

    colors = ['b', 'r', 'g']
    plt.figure(figsize=(7, 5))
    for sampling in samplings:
        plt.plot(accuracies_train_all[nus.index(0.5)][samplings.index(sampling)], c=colors[samplings.index(sampling)], label=f"Train set sampling={sampling}")
        plt.plot(accuracies_test_all[nus.index(0.5)][samplings.index(sampling)], c=colors[samplings.index(sampling)], label=f"Test set sampling={sampling}", linestyle='dashed')
    plt.xlabel('Number of rules')
    plt.ylabel('Accuracy')
    plt.title('Impact of Different Sampling Values on Accuracy', wrap=True)
    plt.legend()
    plt.savefig(os.path.join("plots", "Regularization", 'Regularization comparison - sampling.png'))
    plt.show()

    plt.figure(figsize=(7, 5))
    for nu in nus:
        plt.plot(accuracies_train_all[nus.index(nu)][samplings.index(0.5)], c=colors[nus.index(nu)], label=f"Train set shrinkage={nu}")
        plt.plot(accuracies_test_all[nus.index(nu)][samplings.index(0.5)], c=colors[nus.index(nu)], label=f"Test set shrinkage={nu}", linestyle='dashed')
    plt.xlabel('Number of rules')
    plt.ylabel('Accuracy')
    plt.title('Impact of Different Shrinkage Values on Accuracy', wrap=True)
    plt.legend()
    plt.savefig(os.path.join("plots", "Regularization", 'Regularization comparison - shrinkage.png'))
    plt.show()
