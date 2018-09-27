# digit classifier

from keras.models import load_model
from scipy import misc
from skimage import io, transform
import numpy as np


# vars
model_path = '/Users/ElliottJobson/Documents/cs231a/final_project/cs231a_project/src/deep_model.h5'
train_image_path = '/Users/ElliottJobson/Documents/cs231a/final_project/project_data/English/Fnt/Sample006/img006-00027.png'


# test
img = misc.imread(train_image_path, mode='RGB').astype(dtype=float)
img /= 255
img = transform.resize(img, output_shape=(28, 28, 3))
img = img.dot([0.299, 0.587, 0.114])
img = img.reshape(1, 1, 28, 28).astype('float32')
x = img


"""
Input: numpy array of images of shape (N, 1, 28, 28)
Returns: numpy array of digit predictions of shape (N,)
Usage: from classifier import *
	preds = classify(input_imgs)

"""
def classify(input_imgs=x):

	saved_model = load_model(model_path)
	preds = saved_model.predict(input_imgs)
	preds = np.argmax(preds, axis=1)

	return preds

if __name__ == "__main__":
	classify(x)