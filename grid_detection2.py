import cv2
import numpy as np
from matplotlib import pyplot as plt

def binarize(sudoku, dilate = True):
	outline = np.zeros(sudoku.shape, dtype = np.uint8)
	sudoku = cv2.GaussianBlur(sudoku, (11,11), 0)
	outline = cv2.adaptiveThreshold(sudoku, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
	outline = cv2.bitwise_not(outline)
	kernel = np.matrix([[0,1,0],[1,1,1],[0,1,0]], dtype = np.uint8)
	if (dilate):
		outline = cv2.dilate(outline, kernel)

	return outline

def findLongestContour(binarized_image):
	#Find contours
	contours, hierarchy = cv2.findContours(binarized_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	longest_contour = None
	longest_len = 0

	#for loop ders
	for i in range(len(contours)):
		curr_len = contours[i].shape[0]
		if(curr_len > longest_len):
			longest_contour = contours[i]
			longest_len = curr_len

	return longest_contour

def findCorners(contour_array):
	box_points = contour_array[:,0,:]
	point_sum = box_points[:,0] + box_points[:,1]
	point_diff = box_points[:,0] - box_points[:,1]

	iBR = np.argmax(point_sum)
	iTR = np.argmin(point_sum)
	iBL = np.argmin(point_diff)
	iTL = np.argmax(point_diff)

	br = box_points[iBR]
	tl = box_points[iTL]
	bl = box_points[iBL]
	tr = box_points[iTR]

	return (br, bl, tr, tl)

def longestEdge(br, bl, tr, tl):
	maxLength = (bl[0]-br[0])*(bl[0]-br[0]) + (bl[1]-br[1])*(bl[1]-br[1])
	temp = (tr[0]-br[0])*(tr[0]-br[0]) + (tr[1]-br[1])*(tr[1]-br[1])

	if(temp>maxLength):
		maxLength = temp

	temp = (tr[0]-tl[0])*(tr[0]-tl[0]) + (tr[1]-tl[1])*(tr[1]-tl[1])
	if(temp>maxLength):
		maxLength = temp

	temp = (bl[0]-tl[0])*(bl[0]-tl[0]) + (bl[1]-tl[1])*(bl[1]-tl[1])
	if(temp>maxLength):
		maxLength = temp

	return np.sqrt(maxLength)



def detect_grid(im_path='../images/sudoku_6.jpg'):

	########################################################################
	#TO DO: Code for passing in file arguments (ideally many many files at once)#
	########################################################################

	#binarize image
	sudoku = cv2.imread(im_path, 0)
	# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	# sudoku = clahe.apply(sudoku)

	image = binarize(sudoku)
	image_cp = image.copy()

	print "Started grid detection..."
	#effectively find the bounding box
	bounding_box = findLongestContour(image_cp)

	# find corners
	br, bl, tl, tr = findCorners(bounding_box)

	cv2.circle(image_cp, (br[0],br[1]), 5, (255,255,0), -1)
	cv2.circle(image_cp, (bl[0],bl[1]), 5, (255,255,0), -1)
	cv2.circle(image_cp, (tr[0],tr[1]), 5, (255,255,0), -1)
	cv2.circle(image_cp, (tl[0],tl[1]), 5, (255,255,0), -1)

	black = [0,0,0]
	outline_plot = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
	cont_plot = cv2.copyMakeBorder(image_cp, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)

	maxLength = longestEdge(br, bl, tr, tl)

	src = np.matrix([tl, tr, br, bl], dtype = np.float32)
	dst = np.matrix([[0,0],[maxLength, 0],[maxLength, maxLength], [0, maxLength]], dtype = np.float32)

	fixed = np.matrix((maxLength, maxLength), dtype = np.uint8)
	fixed = cv2.warpPerspective(sudoku, cv2.getPerspectiveTransform(src, dst), (int(maxLength), int(maxLength)))

	fixed = cv2.resize(fixed, (360,360))
	maxLength = 360

	fixed_threshold = cv2.adaptiveThreshold(fixed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 1)

	dist = np.floor(maxLength/9).astype(np.int16)

	currentCell = np.zeros((dist, dist))

	final_img = cv2.hconcat((outline_plot, cont_plot))
	fixed_binarized = binarize(fixed, None)

	edges = cv2.Canny(fixed,50,50, 3)
	lines = cv2.HoughLines(edges,1,np.pi/180,150)
#	print lines
	if lines != None:
		for rho,theta in lines[0]:
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))
	#		print x1, y1, x2, y2
			cv2.line(fixed_threshold, (x1, y1), (x2, y2), (0,0,0), 5)


	cells = []
	locations = []
	for i in range(9):
		for j in range(9):
			currentCell = fixed_threshold[i*dist:(i+1)*dist, j*dist:(j+1)*dist]

			# Elliott temp code: attempt to dilate so that 1's are better seen
		#	kernel = np.matrix([[1,1,1],[1,1,1],[1,1,1]], dtype = np.uint8)
		#	currentCell = cv2.dilate(currentCell, kernel)

##########ders analysis
			corners = cv2.cornerHarris(currentCell[10:30, 10:30], 2, 3, 0.04)
			# # cv2.circle(image_cp, (br[0],br[1]), 5, (255,255,0), -1)
			# plt.imshow(corners ,cmap = 'jet')
			# print(corners.reshape((40*40, 1)))
			# plt.show()
		#	plt.imshow(currentCell)
		#	plt.show()
		#	print(corners[corners != 0].shape)
		#	print(corners[corners != 0].shape[0])
			if (corners[corners != 0].shape[0] > 100):
		#		print(corners.shape)
				locations.append((i,j))
				cells.append(currentCell)
		#		plt.imshow(corners, cmap='jet')
		#		plt.show()
		#		plt.imshow(currentCell)
		#		plt.show()

###########

		# 	w, h = currentCell.shape
		# 	analysis = fixed_threshold[i*dist+(w/5):(i+1)*dist-(w/5), j*dist+(h/5):(j+1)*dist-(h/5)]

		# 	moments = cv2.moments(analysis, True)
		# 	m = moments['m00']
		# #	print m
		# #	print analysis.shape[0]*analysis.shape[1]/6



		# 	if m >= analysis.shape[0]*analysis.shape[1]/6:
		# 		locations.append((i,j))
		# 		cells.append(currentCell)
				#print "yes"
			# else:
			# 	print "no"
			# print "\n"
			# cv2.imshow('Test analysis', analysis)
			# cv2.imshow('Test cell', currentCell)
			# cv2.waitKey(0)
	# cv2.imshow('Final', final_img)
	# cv2.imshow('Rectified', fixed)
	# cv2.imshow('Threshold', fixed_threshold)
	# cv2.imshow('Binarized Rectified Image', fixed_binarized)
	# cv2.imshow('Binarized Threshold Image', fixed_threshold)

	return cells, locations


#	cv2.waitKey(0)
#	cv2.destroyAllWindows()



if __name__ == '__main__':

	detect_grid()












