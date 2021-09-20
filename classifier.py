from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

	#taking clothes data from keras similar to mnsit data.

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure()
plt.imshow(train_images[random.randint(0,60000)])
plt.colorbar()
plt.grid(False)
plt.show()

#heatmap to depict random image

train_images = train_images/255
test_images = test_images/255

#all input values should be betweeen 0 and 1

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#initial simple model, flatten to 784*1 matrix, then down to 128 nodes and then to the number of cloth classes.

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=5)

#multiple epochs needed to cover the whole gradient descent curve

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

#checking accuracy on test data, a lower value indicates overfitting

predictions = model.predict(test_images)

sample_image_num = random.randint(0,10000)
plt.figure()
plt.imshow(test_images[sample_image_num])
plt.colorbar()
plt.grid(False)
plt.show()
class_names[np.argmax(predictions[sample_image_num])]

#shows a coloured discription of a cloth item and then predicts its type

	
