from sklearn.metrics import accuracy_score, mean_absolute_error
import numpy as np


def calculate_accuracy(y, y_preds):
    y_pred_labels = [np.argmax(y_pred) for y_pred in y_preds]
    return accuracy_score(y, y_pred_labels)


def calculate_all_metrics(y, y_preds):

    accuracy = calculate_accuracy(y, y_preds)
    mean_absolute_err = None

    return {'accuracy': accuracy, 'mean_absolute_error': mean_absolute_err}
