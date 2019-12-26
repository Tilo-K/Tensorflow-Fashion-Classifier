**Tensorflow Example**

This is an example Project which trains a model in [_Tensorflow 2.0_]("https://www.tensorflow.org/" "the Tensorflow Homepage")<br/>
The model will be trained to recognise different type of clothes such as: <br />
`class names =` <br/>
`['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']` <br />

The data is based on the [Fashion-MNIST Dataset](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/)
by Kashif Rasul & Han Xiao.<br />

Certain things can be tweaked to change and maybe improve the model. <br /> <br />
_**For example:**_ <br />

The number and settings of the layers: <br />
`model = keras.Sequential([` <br />
    `keras.layers.Flatten(input_shape=(28, 28)),   # Input Layer`  <br />
    `keras.layers.Dense(128, activation='relu'),   # Hidden Layer` <br />
    `keras.layers.Dense(128, activation='relu'),   # Hidden Layer` <br />
    `keras.layers.Dense(10, activation='softmax')  # Output Layer` <br />
`])` <br />

Or the number of epochs the model is goint to be trained: <br />
`model.fit(train_images, train_labels, epochs=5)`