import pickle
from sklearn.model_selection import train_test_split


from EnderClassifier import EnderClassifier
from PrepareDatasets import prepare_wine_classification_dataset, prepare_apple_classification_dataset, prepare_apple_small_classification_dataset, prepare_smallest_classification_dataset
from CalculateMetrics import calculate_all_metrics
from VisualiseHistory import visualise_history
import numpy as np

n_rules = 200
use_gradient = True
save_history = True
# optimized_searching_for_cut = True
optimized_searching_for_cut = False
prune = False
# TRAIN_NEW = False
TRAIN_NEW = True

# X, y = prepare_wine_classification_dataset()
# X, y = prepare_apple_small_classification_dataset()
X, y = prepare_apple_classification_dataset()
# X, y = prepare_smallest_classification_dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


if TRAIN_NEW:
    ender = EnderClassifier(n_rules=n_rules, use_gradient=use_gradient, save_history=save_history, optimized_searching_for_cut=optimized_searching_for_cut, prune=prune)
    ender.fit(X_train, y_train)
    with open (f'model_{n_rules}.pkl', 'wb') as f:
        pickle.dump(ender, f, pickle.HIGHEST_PROTOCOL)
else:
    with open (f'model_{n_rules}.pkl', 'rb') as f:
        ender = pickle.load(f)
    # ender.prune_rules(alpha=10e10) # n_rules
    ender.prune_rules(alpha=0.05)



y_train_preds = ender.predict(X_train, dont_use_effective_rules=True)
y_test_preds = ender.predict(X_test, dont_use_effective_rules=True)
final_metrics = calculate_all_metrics(y_train, y_train_preds, y_test, y_test_preds)
print("Before pruning:", final_metrics)

y_train_preds = ender.predict(X_train)
y_test_preds = ender.predict(X_test)
final_metrics = calculate_all_metrics(y_train, y_train_preds, y_test, y_test_preds)
print("After pruning:", final_metrics)

# visualise_history(ender)
print(f"Normal accuracy when using same no. rules: {ender.history['accuracy'][len(ender.effective_rules)]}")
print(ender.history['accuracy'])
