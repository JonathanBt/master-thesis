import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

def evaluation(model, X_test, Y_test):
    """This function creates the classification report and confusion matrices for each label.

    Args:
        model (fitted model): Fitted model.
        X_test (np.array): Test set examples.
        Y_test (np.array): Test set labels.

    Returns:
        None.
    """

    # Compute predictions
    predictions = model.predict(X_test)

    # Print classification report
    print(classification_report(Y_test, predictions))

    # Plot confusion matrices for each label
    sdgs = [1, 3, 4, 5, 6, 7, 8, 11, 12, 13, 15, 16, 17]
    counter = 0
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 22))
    plt.subplots_adjust(hspace=0.5)
    for ax in axes.flatten():
        if counter <= 12:
            cm = confusion_matrix(Y_test[:, counter], predictions[:, counter])
            sns.heatmap(cm, annot=True, fmt="g", ax=ax, cmap="Greens")
            ax.set_title("SDG " + str(sdgs[counter]))
            ax.set_xlabel("Predicted labels")
            ax.set_ylabel("True labels")
            counter += 1
        else:
            ax.set_visible(False)
            counter += 1