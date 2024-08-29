import matplotlib.pyplot as plt
import os


def visualise_history(model):
    plt.figure(figsize=(10, 7))
    plt.plot([i for i in range(model.n_rules + 1)], model.history['accuracy'], label='Train dataset')
    if model.X_test is not None:
        plt.plot([i for i in range(model.n_rules + 1)], model.history['accuracy_test'], label='Test dataset')
    plt.legend()
    plt.grid()
    plt.ylabel("Accuracy")
    plt.xlabel("Rules")
    plt.title("Accuracy vs no. rules - normal training")
    plt.savefig(os.path.join('Plots',
                             'Training',
                             f'Model_{model.dataset_name}_{model.n_rules}.png'))
    plt.show()
