# game logic for our sudoku solver

"""
input: 9x9 numpy array


"""

import numpy as np

# instance var for possible values (will eliminate these as we solve the game)
y = range(9)
vals = range(1,10,1)
num_squares = 81

pos_values_arr = np.array([
	[np.array(vals) for _ in y],
	[np.array(vals) for _ in y],
	[np.array(vals) for _ in y],
	[np.array(vals) for _ in y],
	[np.array(vals) for _ in y],
	[np.array(vals) for _ in y],
	[np.array(vals) for _ in y],
	[np.array(vals) for _ in y],
	[np.array(vals) for _ in y],
	])

# initialize all vals to -1 to indicate vals haven't been solved
def setup_board():

	solved_vals = np.ones((9,9), dtype=int) * -1 # (h, w)
	solved_vals[0][5] = 4
	solved_vals[0][7] = 2
	solved_vals[0][8] = 8

	solved_vals[1][0] = 4
	solved_vals[1][2] = 6
	solved_vals[1][8] = 5

	solved_vals[2][0] = 1
	solved_vals[2][4] = 3
	solved_vals[2][6] = 6

	solved_vals[3][3] = 3
	solved_vals[3][5] = 1

	solved_vals[4][1] = 8
	solved_vals[4][2] = 7
	solved_vals[4][6] = 1
	solved_vals[4][7] = 4

	solved_vals[5][3] = 7
	solved_vals[5][5] = 9

	solved_vals[6][2] = 2
	solved_vals[6][4] = 1
	solved_vals[6][8] = 3

	solved_vals[7][0] = 9
	solved_vals[7][6] = 5
	solved_vals[7][8] = 7

	solved_vals[8][0] = 6
	solved_vals[8][1] = 7
	solved_vals[8][3] = 4

	return solved_vals


def calc_num_solved(solved_vals):
	ret = 0
	for i in range(9):
		for j in range(9):
			if (solved_vals[i,j] != -1):
				ret += 1
	return ret


"""
Function updates the pos_values_arr, given a newly solved cell.
pos_values_array: array of possible values for each cell in the grid
new_val: the solved value at the location specified by coordinates
coordinates: tuple (h,w) specifying the location of a recently-solved cell

"""
def update_vertical(pos_values_arr, new_val, coordinates):

	row_index = coordinates[1]

	for i in range(9):
		if i is not coordinates[0]:

			pos_cell_vals = pos_values_arr[i,row_index]

			# this won't work, return value is whack for numpy list types
			inds_to_delete = np.where(pos_cell_vals == new_val)





def update_pos_vals(pos_values_arr, solved_vals, recently_solved_list):

	for coords in recently_solved_list:

		new_val = solved_vals[coords]
		update_vertical(pos_values_arr, new_val, coordinates)

		update_horizontal(pos_values_arr, new_val, coordinates)

		update_sections(pos_values_arr, new_val, coordinates)

	# returns nothing

"""
Where there is only one possible value, update the solved values
"""
def update_solved_vals(pos_values_arr, solved_vals):

	num_updated = 0

	for i in range(9):
		for j in range(9):
			if (len(pos_values_arr[i,j]) == 1):
				solved_vals[i,j] = pos_values_arr[i,j][0]
				num_updated += 1

	return num_updated




if __name__ == "__main__":

	solved_vals = setup_board()

	num_solved = calc_num_solved(solved_vals)

	while (num_solved < num_squares):

		# might want to instead pass in a list of tuples of recently solved vals
		update_pos_vals(pos_values_arr, solved_vals)

		# returns number of updated values
		num_solved += update_solved_vals(pos_values_arr, solved_vals)

	return solved_vals







