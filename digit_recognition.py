# digit recognition

import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

model_save_path = '/Users/ElliottJobson/Documents/cs231a/final_project/cs231a_project/src/feed_forward_model.h5'


# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flatten 28 x 28 images
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalize inputs (grayscale vals)
X_train /= 255
X_test /= 255

# create one-hot encodings of the outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define the baseline model

def baseline_model():
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, 
		kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, 
		kernel_initializer='normal', activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam',
		metrics=['accuracy'])
	return model

# run it
model = baseline_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test),
	epochs=10, batch_size=200, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
model.save(model_save_path)



# test loading of images / mnist dataset

"""
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))

plt.show()


# baseline model (not the deep CNN)

"""