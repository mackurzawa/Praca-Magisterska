import matplotlib.pyplot as plt


def visualise_history(model):
    plt.figure(figsize=(10, 7))
    plt.plot([i for i in range(model.n_rules + 1)], model.history['accuracy'])
    plt.grid()
    plt.ylabel("Accuracy")
    plt.xlabel("Rules")
    plt.title("Accuracy vs no. rules")
    plt.show()
