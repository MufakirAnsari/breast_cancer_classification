import joblib
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def train_svm(X_train, y_train, model_path="svm_model.pkl"):
    """
    Train a Support Vector Machine model and save it.
    """
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X_train.reshape(X_train.shape[0], -1), np.argmax(y_train, axis=1))
    joblib.dump(clf, model_path)
    print(f"SVM model saved to {model_path}")
    return clf


def load_svm(model_path="svm_model.pkl"):
    """
    Load a pre-trained SVM model.
    """
    clf = joblib.load(model_path)
    print(f"SVM model loaded from {model_path}")
    return clf


def evaluate_svm(model, X_test, y_test):
    """
    Evaluate the SVM model on the test dataset.
    """
    y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    print(f"SVM Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(np.argmax(y_test, axis=1), y_pred))
    return accuracy