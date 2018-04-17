import torch
from torch.autograd import Variable


def convert_to_quadruple(prog_input_output, lam_or_for):
	#the prog_tree code is a placeholder depending on how different make_tree for the two
	#end up being and what we call them
	if lam_or_for:
		prog_tree = lam_make_tree(prog_input_output[0])
	else:
		prog_tree = for_make_tree(prog_input_output[0])
	input_matrix = torch.zeros(10, 10)
	input_matrix[0][int(prog_input_output[1][0])] = 1
	input_matrix[1:9] = 0.1
	output_matrix = torch.zeros(10, 10)
	input_matrix[0][int(prog_input_output[1][1])] = 1
	input_matrix[1:9] = 0.1
	mask = torch.zeros(10, 10)
	mask[0] = 1
	return prog_tree, input_matrix, output_matrix, mask