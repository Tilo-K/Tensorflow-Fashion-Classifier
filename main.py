import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
from random import randint

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()  # Load and split-up the dataset

class_names = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# Creating the model with the different layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),   # Input Layer
    keras.layers.Dense(128, activation='relu'),   # Hidden Layer
    keras.layers.Dense(128, activation='relu'),   # Hidden Layer
    keras.layers.Dense(10, activation='softmax')  # Output Layer
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)  # Train the model for 5 epochs (In this case 5 are enough)
test_loss, test_acc = model.evaluate(test_images, test_labels)  # Test the model on none training data

print('Accuracy: ', test_acc)

# Save the model if requested
save = input('Save this model ? (y/n)')

if 'y' in save:
    name = 'fashion_model_' + str(test_acc) + '_' + str(time.time()) + '.h5'
    model.save(name)
    print('Saved as:\t' + name)

# Manually preview and check random results
save = input('Preview ? (y/n)')

if 'y' in save:
    start = randint(0, len(test_images)-5)
    print('Preview of ' + str(start) + '-' + str(start+5))
    prediction = model.predict(test_images)

    for i in range(start, start+6):
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel('Actual: ' + class_names[test_labels[i]])
        plt.title('Prediction:' + class_names[np.argmax(prediction[i])])
        plt.show()
