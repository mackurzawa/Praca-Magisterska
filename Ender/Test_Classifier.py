import pickle

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
TRAIN_NEW = False


# X, y = prepare_wine_classification_dataset()
# X, y = prepare_apple_small_classification_dataset()
X, y = prepare_apple_classification_dataset()
# X, y = prepare_smallest_classification_dataset()

if TRAIN_NEW:
    ender = EnderClassifier(n_rules=n_rules, use_gradient=use_gradient, save_history=save_history, optimized_searching_for_cut=optimized_searching_for_cut, prune=prune)
    ender.fit(X, y)
    with open ('model.pkl', 'wb') as f:
        pickle.dump(ender, f, pickle.HIGHEST_PROTOCOL)
else:
    with open ('model.pkl', 'rb') as f:
        ender = pickle.load(f)
    ender.prune_rules(alpha=0.01)



y_preds = ender.predict(X, show_before_pruning=True)
final_metrics = calculate_all_metrics(y, y_preds)
print("Before pruning:", final_metrics)
y_preds = ender.predict(X)
final_metrics = calculate_all_metrics(y, y_preds)
print("After pruning:", final_metrics)

# visualise_history(ender)
print(f"Normal accuracy when using same no. rules: {ender.history['accuracy'][len(ender.effective_rules)]}")

