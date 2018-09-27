
# Overall pipeline for solving Sudoku puzzle

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import transform

# our own imports
from grid_detection import *
from digit_classifier import *



# vars
im_height = 28
im_width = 28

"""
Takes as input a list of images, one for each cell. Reshapes them and puts them into
one numpy array of shape (N, 1, 28, 28). Returns this array
"""
def reshape_cells(cells): # input typically (N, 40, 40) ... however, it doesn't matter, as long as first dim is N

	cell_arr = np.asarray(cells)
	N = cell_arr.shape[0]
	cell_arr = transform.resize(cell_arr, output_shape=(N, im_height, im_width)) # (N, 28, 28)
	cell_arr = np.expand_dims(cell_arr, axis=1) # (N, 1, 28, 28)
	cell_arr = 1. - cell_arr # invert cell_arr to have black font on a white background
	return cell_arr 



if __name__ == "__main__":
	
	#binarize image
	sudoku = cv2.imread('../images/sudoku_5.jpg', 0)
	image = binarize(sudoku)
	image_cp = image.copy()

	#find the bounding box
	bounding_box = findLongestContour(image_cp)

	# find corners
	br, bl, tr, tl = findCorners(bounding_box)
	maxLength = longestEdge(br, bl, tr, tl)

	cell_size = np.floor(maxLength/9).astype(np.int16)
	fixed = rectify(sudoku, br, bl, tr, tl, maxLength)

	##########GIVES MORE ROBUST RESULTS. EXTRA STEP IN THE PIPELINE############
	fixed_threshold = cv2.adaptiveThreshold(fixed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 1)
	fixed_binarized = binarize(fixed, None)
	###########################################################################

	# extract the data we need to solve the puzzle
	currentCell = np.zeros((cell_size, cell_size))
	cells, locations = cells_from_image(fixed_threshold, cell_size)


	# just testing....
	for cell in cells:
		cv2.imshow('cell', cell)
		cv2.waitKey(0)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	

	# (N, 40, 40) -> (N, 1, 28, 28)
	cells = reshape_cells(cells)
	digit_list = classify(cells)
	print(digit_list)





