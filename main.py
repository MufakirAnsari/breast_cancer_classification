from data_loader import load_image_paths, split_images_by_class
from data_preprocessor import data_generator
from model_builder import build_cnn_model
from model_trainer import train_model, evaluate_model, plot_training_history
from svm_model import train_svm, evaluate_svm, load_svm
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from memory_profiler import profile

# Constants
BASE_DIRECTORY = r"C:\Users\mufak\Desktop\Github\breast_cancer_classification\archive\IDC_regular_ps50_idx5"

if __name__ == "__main__":
    # Step 1: Load Image Paths
    image_paths = load_image_paths(BASE_DIRECTORY)
    print(f"Total images found: {len(image_paths)}")

    # Step 2: Split Images by Class
    non_cancer_images, cancer_images = split_images_by_class(image_paths)
    print(f"Non-cancer images: {len(non_cancer_images)}, Cancer images: {len(cancer_images)}")

    # Validate that we have data for both classes
    if len(non_cancer_images) == 0 or len(cancer_images) == 0:
        raise ValueError("No images found for one or both classes.")

    # Combine and shuffle data
    all_images = non_cancer_images + cancer_images
    labels = [0] * len(non_cancer_images) + [1] * len(cancer_images)

    # Convert labels to categorical
    labels = tf.keras.utils.to_categorical(labels, num_classes=2)

    # Split into training and testing sets
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        all_images, labels, test_size=0.2, random_state=42
    )

    # Print shapes to verify
    print("X_train_paths shape:", len(X_train_paths))
    print("y_train shape:", y_train.shape)
    print("X_test_paths shape:", len(X_test_paths))
    print("y_test shape:", y_test.shape)

    # Step 3: Build CNN Model
    input_shape = (50, 50, 3)
    cnn_model = build_cnn_model(input_shape)

    # Step 4: Train Model Using Data Generator
    batch_size = 32
    train_gen = data_generator(X_train_paths, y_train, batch_size=batch_size)
    test_gen = data_generator(X_test_paths, y_test, batch_size=batch_size)

    steps_per_epoch = len(X_train_paths) // batch_size
    validation_steps = len(X_test_paths) // batch_size

    @profile
    def train_with_memory_profile():
        history = cnn_model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=test_gen,
            validation_steps=validation_steps,
            epochs=25
        )
        return history

    history = train_with_memory_profile()

    # Step 5: Evaluate Model
    evaluate_model(cnn_model, test_gen, validation_steps)
    plot_training_history(history)