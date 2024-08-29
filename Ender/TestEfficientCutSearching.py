from time import time
import random


from sklearn.model_selection import StratifiedKFold

from EnderClassifier import EnderClassifier
from EnderClassifierFastMyImplementation import EnderClassifierFastMyImplementation
from PrepareDatasets import prepare_dataset

def test_time(model, RANDOM_STATE):
    times = []
    accuracies_train = []
    accuracies_test = []
    for i_crossval, (train_indices, test_indices) in enumerate(StratifiedKFold(n_splits=KFolds, shuffle=True, random_state=RANDOM_STATE).split(X, y)):
        X_train, y_train, X_test, y_test = X.iloc[train_indices], y[train_indices], X.iloc[test_indices], y[test_indices]


        time_started = time()
        model.fit(X_train, y_train, X_test=X_test, y_test=y_test)
        time_elapsed = round(time() - time_started, 2)


        model.evaluate_all_rules()


        times.append(time_elapsed)
        accuracies_train.append(model.history['accuracy'][-1])
        accuracies_test.append(model.history['accuracy_test'][-1])
    return times


if __name__ == "__main__":
    n_rules = 100
    use_gradient = True
    # use_gradient = False
    optimized_searching_for_cut = 0
    optimized_searching_for_cut = 1
    prune = False
    # TRAIN_NEW = False
    TRAIN_NEW = True
    # dataset = 'apple'
    # dataset = 'wine'
    ##########
    # dataset = 'haberman'
    dataset = 'breast-c'
    # dataset = 'spambase'

    nu = .5
    # sampling = .5
    sampling = 1

    params = {
        "Classification": True,
        "Dataset": dataset,
        "n_rules": n_rules,
        "Gradient. not Newton_Raphson": use_gradient,
        "Shrinkage. nu": nu,
        "Sampling. percentage": sampling,
    }
    random.seed(42)
    KFolds = 5
    X, y = prepare_dataset(dataset)
    times_ender = []
    times_ender_fast = []
    for RANDOM_STATE in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        ender = EnderClassifier(dataset_name=dataset, n_rules=n_rules, use_gradient=use_gradient,
                                optimized_searching_for_cut=optimized_searching_for_cut, nu=nu, sampling=sampling,
                                random_state=RANDOM_STATE, verbose=0)
        ender_fast = EnderClassifierFastMyImplementation(dataset_name=dataset, n_rules=n_rules, use_gradient=use_gradient,
                                                     optimized_searching_for_cut=optimized_searching_for_cut, nu=nu,
                                                       sampling=sampling, random_state=RANDOM_STATE, verbose=0)

        print("ENDER", RANDOM_STATE)
        time_ender = test_time(ender, RANDOM_STATE)
        print("ENDER_FAST", RANDOM_STATE)
        time_ender_fast = test_time(ender_fast, RANDOM_STATE)
        print(time_ender)
        print(time_ender_fast)
        times_ender += time_ender
        times_ender_fast += time_ender_fast
    print(times_ender)
    print(times_ender_fast)

