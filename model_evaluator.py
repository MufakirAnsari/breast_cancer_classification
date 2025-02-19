import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test dataset.
    """
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    return loss, accuracy


def plot_confusion_matrix(model, X_test, y_test):
    """
    Plot the confusion matrix for the test dataset.
    """
    Y_pred = model.predict(X_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(y_test, axis=1)

    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    plt.figure(figsize=(8, 8))
    sns.heatmap(confusion_mtx, annot=True, fmt='.1f', cmap="BuPu", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(Y_true, Y_pred_classes))


def plot_training_history(history):
    """
    Plot training and validation accuracy/loss over epochs.
    """
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')