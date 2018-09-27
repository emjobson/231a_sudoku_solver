# deep convolutional network for digit recognition

import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# new imports for chars74k stuff
import pandas as pd
from skimage import io, transform
from scipy import misc
import matplotlib.pyplot as plt
K.set_image_dim_ordering('th')

# paths
model_save_path = '/Users/ElliottJobson/Documents/cs231a/final_project/cs231a_project/src/deep_model.h5'
train_image_path = '/Users/ElliottJobson/Documents/cs231a/final_project/project_data/English/Fnt/Sample'

# vars
im_height = 28
im_width = 28
chars74k_height = 128
chars74k_width = 128
num_chars74_cat = 1016
num_train_74k = num_chars74_cat * 10
num_train = 60000
num_classes = 10


# setup the convolutional model, return model for usage
def deep_conv_model():

	model = Sequential()
	model.add(Conv2D(30, (5,5), input_shape=(1,28,28),
		activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(15, (3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam',
		metrics=['accuracy'])

	return model


def setup():

	# set up MNIST data
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
	X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

	# normalize and invert MNIST images
	X_train /= 255 # (N, 1, 28, 28)
	X_test /= 255
	X_train = 1. - X_train # invert colors, so we have black letters on a white background
	X_test = 1. - X_test # invert colors, so we have black letters on a white background


	y_train = np_utils.to_categorical(y_train) # create one-hot encodings
	y_test = np_utils.to_categorical(y_test)
	num_classes = y_test.shape[1]

	# set up Chars74K data
	y_train_74k = []
	x_train_74k = np.empty(shape=(num_train_74k, im_height, im_width, 3))


	print('Starting to resize and save chars74k images...')
	counter = 0
	for i in range(10): #[0,9] ... iterator over numbers

		num = str(i+1).zfill(3)
		
		for pic_index in range(num_chars74_cat):

			pic_index = str(pic_index+1).zfill(5)

			image = misc.imread(train_image_path + num + '/img' 
				+ num + '-' + pic_index + '.png', mode='RGB').astype(dtype=float)

			image /= 255 # currently (128, 128, 3)

			image = transform.resize(image, output_shape=(im_height, im_width, 3)) # (28, 28, 3)

			x_train_74k[counter] = image

			counter += 1

		num_list = [i] * num_chars74_cat
		y_train_74k.extend(num_list)


	x_train_74k = np.tensordot(x_train_74k, np.array([0.299, 0.587, 0.114]).reshape((3,1)), axes=1) # (N, 128, 128, 1)
	x_train_74k = x_train_74k.reshape(x_train_74k.shape[0], 1, im_height, im_width).astype('float32') # (N, 1, 28, 28)

	y_train_74k = np.asarray(y_train_74k, dtype=int)
	y_train_74k = np_utils.to_categorical(y_train_74k) # create one-hot encoding

	x_train_all = np.vstack((X_train, x_train_74k))
	y_train_all = np.vstack((y_train, y_train_74k))

	shuffle_inds = np.random.permutation(x_train_all.shape[0])

	x_train_all = x_train_all[shuffle_inds]
	y_train_all = y_train_all[shuffle_inds]

	# separate train and test data
	x_train_final = x_train_all[:num_train]
	y_train_final = y_train_all[:num_train]

	x_test_final = x_train_all[num_train:] 
	y_test_final = y_train_all[num_train:]

	return x_train_final, y_train_final, x_test_final, y_test_final


if __name__ == "__main__":

	# preprocess the dataset
	x_train_final, y_train_final, x_test_final, y_test_final = setup()

	# run the model
	model = deep_conv_model()
	"""
	print(model.summary())
	quit()
	"""

	hist = model.fit(x_train_final, y_train_final, validation_data=(x_test_final, y_test_final),
		epochs=10, batch_size=200)
	scores = model.evaluate(x_test_final, y_test_final, verbose=0)
	print("Deep Conv Model Error: %.2f%%" % (100-scores[1]*100))
	model.save(model_save_path)

	# plot the model's accuracy
	plt.plot(hist.history['acc'])
	plt.plot(hist.history['val_acc'])
	plt.title('Model Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

	# plot the model's loss
	plt.plot(hist.history['loss'])
	plt.plot(hist.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

