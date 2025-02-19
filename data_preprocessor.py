import cv2
import numpy as np
import random
from tensorflow.keras.utils import to_categorical

def preprocess_image(image_path, label, target_size=(50, 50)):
    """
    Preprocess a single image by resizing and normalizing.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        return resized_img, label
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def data_generator(image_paths, labels, batch_size=32, target_size=(50, 50)):
    """
    A generator to yield batches of preprocessed images and labels.
    """
    num_samples = len(image_paths)
    while True:
        indices = np.random.permutation(num_samples)  # Shuffle indices
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_images = []
            batch_labels = []

            for idx in batch_indices:
                result = preprocess_image(image_paths[idx], labels[idx], target_size)
                if result is not None:
                    img, label = result
                    batch_images.append(img)
                    batch_labels.append(label)

            if batch_images:
                batch_images = np.array(batch_images, dtype='float32') / 255.0
                batch_labels = np.array(batch_labels)
                yield batch_images, batch_labels