from sklearn.metrics import accuracy_score, mean_absolute_error
import numpy as np


def calculate_accuracy(y, y_preds):
    y_pred_labels = [np.argmax(y_pred) for y_pred in y_preds]
    return accuracy_score(y, y_pred_labels)



def calculate_all_metrics(y_train, y_train_preds, y_test=None, y_test_preds=None):

    # accuracy = accuracy_score(y, y_pred_labels)
    accuracy_train = calculate_accuracy(y_train, y_train_preds)
    accuracy_test = None
    if y_test is not None and y_test_preds is not None:
        accuracy_test = calculate_accuracy(y_test, y_test_preds)



    # y_pred_for_true_labels = [y_preds[i][y[i]] for i in range(len(y))]
    # perfect_probabilites = [1 for _ in y]
    # mean_absolute_err = mean_absolute_error(perfect_probabilites, y_pred_for_true_labels)
    mean_absolute_err = None

    return {'accuracy_train': accuracy_train, 'accuracy_test': accuracy_test, 'mean_absolute_error': mean_absolute_err}
