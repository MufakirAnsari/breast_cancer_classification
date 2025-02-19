from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import ResNet50

def build_cnn_model(input_shape):
    """
    Build a Convolutional Neural Network (CNN) model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        MaxPooling2D(strides=2),
        Dropout(0.25),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((3, 3), strides=2),
        Dropout(0.25),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((3, 3), strides=2),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_resnet_model(input_shape, num_classes):
    """
    Build a ResNet model.
    """
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model