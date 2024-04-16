from sklearn.metrics import accuracy_score, mean_absolute_error
import numpy as np


def calculate_all_metrics(y, y_preds):
    y_pred_labels = [np.argmax(y_pred) for y_pred in y_preds]
    y_pred_for_true_labels = [y_preds[i][y[i]] for i in range(len(y))]
    perfect_probabilites = [1 for _ in y]

    accuracy = accuracy_score(y, y_pred_labels)
    mean_absolute_err = mean_absolute_error(perfect_probabilites, y_pred_for_true_labels)
    mean_absolute_err = None

    return {'accuracy': accuracy, 'mean_absolute_error': mean_absolute_err}
