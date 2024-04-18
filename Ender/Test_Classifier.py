from EnderClassifier import EnderClassifier
from PrepareDatasets import prepare_wine_classification_dataset, prepare_apple_classification_dataset, prepare_apple_small_classification_dataset, prepare_smallest_classification_dataset
from CalculateMetrics import calculate_all_metrics
from VisualiseHistory import visualise_history

n_rules = 2
use_gradient = True
save_history = True


# X, y = prepare_wine_classification_dataset()
X, y = prepare_apple_small_classification_dataset()
# X, y = prepare_apple_classification_dataset()
# X, y = prepare_smallest_classification_dataset()

ender = EnderClassifier(n_rules=n_rules, use_gradient=use_gradient, save_history=save_history)
ender.fit(X, y)

# visualise_history(ender)
y_preds = ender.predict(X)
final_metrics = calculate_all_metrics(y, y_preds)
