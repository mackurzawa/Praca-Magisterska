import pickle
from sklearn.model_selection import train_test_split


from EnderClassifier import EnderClassifier
from PrepareDatasets import prepare_wine_classification_dataset, prepare_apple_classification_dataset, prepare_apple_small_classification_dataset, prepare_smallest_classification_dataset
from CalculateMetrics import calculate_all_metrics
from VisualiseHistory import visualise_history
import numpy as np

n_rules = 20
use_gradient = True
save_history = True
# optimized_searching_for_cut = True
optimized_searching_for_cut = False
prune = False
# TRAIN_NEW = False
TRAIN_NEW = True
dataset = 'apple'
# dataset = 'wine'

# regressor = 'LogisticRegressionL1'
# regressor = 'LogisticRegressionL2'
# regressor = 'MultiOutputRidge'

if dataset == 'wine':
    X, y = prepare_wine_classification_dataset()
# X, y = prepare_apple_small_classification_dataset()
elif dataset == 'apple':
    X, y = prepare_apple_classification_dataset()
# X, y = prepare_smallest_classification_dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


if TRAIN_NEW:
    ender = EnderClassifier(n_rules=n_rules, use_gradient=use_gradient, save_history=save_history, optimized_searching_for_cut=optimized_searching_for_cut, prune=prune)
    ender.fit(X_train, y_train, X_test=X_test, y_test=y_test)
    with open (f'model_{dataset}_{n_rules}.pkl', 'wb') as f:
        pickle.dump(ender, f, pickle.HIGHEST_PROTOCOL)
else:
    with open (f'model_{dataset}_{n_rules}.pkl', 'rb') as f:
        ender = pickle.load(f)
    # ender.prune_rules(alpha=10e10) # n_rules
    # ender.prune_rules(regressor='LogisticRegressionL1', alpha=0.005)
ender.dataset_name = dataset


y_train_preds = ender.predict(X_train, use_effective_rules=False)
y_test_preds = ender.predict(X_test, use_effective_rules=False)
final_metrics_train = calculate_all_metrics(y_train, y_train_preds)
final_metrics_test = calculate_all_metrics(y_test, y_test_preds)
print("Before pruning train:", final_metrics_train)
print("Before pruning test: ", final_metrics_test)

# for pruning_regressor, alpha in [('LarsPath', 1), ('LogisticRegressorL1', 0.005), ('LogisticRegressorL2', 10e-7)]:
pruning_methods = [('LarsPath', 1)]
pruning_methods = [('Wrapper', None)] # Potentially 'accuracy'
# pruning_methods = []
for pruning_regressor, alpha in pruning_methods:
    print()
    print(pruning_regressor)
    ender.prune_rules(pruning_regressor, alpha=alpha, x_tr=X_train, x_te=X_test, y_tr=y_train, y_te=y_test, lars_how_many_rules=1, lars_show_path=True, lars_show_accuracy_graph=True, lars_verbose=True)
    y_train_preds = ender.predict(X_train)
    y_test_preds = ender.predict(X_test)
    final_metrics_train = calculate_all_metrics(y_train, y_train_preds)
    final_metrics_test = calculate_all_metrics(y_test, y_test_preds)
    # print("After pruning:", final_metrics)
    #
    # print(f"Normal accuracy when using same no. rules: {ender.history['accuracy'][len(ender.effective_rules)]}")

# visualise_history(ender)

# # Predicting with specific rules
# rules_indices = [0, 1]
# my_preds = ender.predict_with_specific_rules(X_train, rules_indices)
# print(calculate_all_metrics(y_train, my_preds))
