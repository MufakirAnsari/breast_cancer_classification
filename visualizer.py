import matplotlib.pyplot as plt
import numpy as np
def predict_and_visualize(model, X_test, y_test, index=0):
    """
    Predict and visualize a single test image.
    """
    input_image = X_test[index:index+1]
    prediction = model.predict(input_image)[0].argmax()
    true_label = np.argmax(y_test[index])

    plt.title(f'True: {true_label}, Predicted: {prediction}')
    plt.imshow(input_image[0])
    plt.axis('off')
    plt.show()