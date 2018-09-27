# recursive sudoku solver

import numpy as np
from timeit import default_timer as timer



# global vars
num_cells = 81

test_vals = np.zeros((9,9), dtype=int) # (h, w)
test_vals[0][5] = 4
test_vals[0][7] = 2
test_vals[0][8] = 8

test_vals[1][0] = 4
test_vals[1][2] = 6
test_vals[1][8] = 5

test_vals[2][0] = 1
test_vals[2][4] = 3
test_vals[2][6] = 6

test_vals[3][3] = 3
test_vals[3][5] = 1

test_vals[4][1] = 8
test_vals[4][2] = 7
test_vals[4][6] = 1
test_vals[4][7] = 4

test_vals[5][3] = 7
test_vals[5][5] = 9

test_vals[6][2] = 2
test_vals[6][4] = 1
test_vals[6][8] = 3

test_vals[7][0] = 9
test_vals[7][6] = 5
test_vals[7][8] = 7

test_vals[8][0] = 6
test_vals[8][1] = 7
test_vals[8][3] = 4


def get_test_cell(board):
	for h in range(9):
		for w in range(9):
			if board[h,w] == 0:
				return (h,w)



def is_legal(test_val, test_cell, board):

	# test if val is legal in row
	if test_val in board[test_cell[0],:]:
		return False

	# test if val is legal in column
	if test_val in board[:, test_cell[1]]:
		return False

	# test if val is legal in square
	sq_h_start = test_cell[0] / 3 * 3
	sq_w_start = test_cell[1] / 3 * 3
	section = board[sq_h_start:sq_h_start+3, sq_w_start:sq_w_start+3]
	if test_val in section:
		return False
	

	return True

def has_solution(board):

	# if board is full
	if np.count_nonzero(board) == num_cells:
		print("We've solved the board!")
		return True

	else:
		test_cell = get_test_cell(board) # find coordinates of zero'd cell
		test_val = 1
		sol_found = False
		while (not sol_found) and (test_val <= 9):

			if is_legal(test_val, test_cell, board):
				board[test_cell] = test_val
				if has_solution(board):
					sol_found = True
					return True
				else:
					board[test_cell] = 0
			test_val += 1

	return sol_found


def find_start_coords(board):

	start_coords = []
	for i in range(9):
		for j in range(9):
			if board[i,j] != 0:
				start_coords.append((i,j))

	return start_coords


"""
Takes in a (9,9) numpy array of ints as input, where unsolved values
are represented as 0's. Uses recursion to solve and print the board if possible.
Indicates that the board was impossible to solve otherwise

Returns (if solution is found): Numpy array with solved values, list of starting coordinates.
List of start coordinates is used so we DON'T superimpose our solved values at those locations.
"""
def solve_board(board=test_vals):

	print(board)
	start = timer()

	start_coords = find_start_coords(board)
#	print 'start_coords', start_coords # uncomment this to see the location of starting board vals

	if (np.array_equal(board, test_vals)):
		print('Using test values for the Sudoku board.')

	if has_solution(board):
		print(board)
		end = timer()
		print(end-start) # uncomment this to see timer
		return board, start_coords

	else:
		print('This board has no solution.')
		end=timer()
		return board, start_coords
#		print(end-start) # uncomment this to see timer



if __name__ == "__main__":

	solve_board()




