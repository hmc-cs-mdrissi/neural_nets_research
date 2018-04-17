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
    return prog_tree, input_matrix, output_matrix, 

class TreeANCDataset(Dataset):
    def __init__(self, path, is_lambda_calculus, num_vars = 10, num_ints = 11, binarize = False, 
                 input_eos_token=True, input_as_seq=False, use_embedding=False,
                 long_base_case=True):
        if is_lambda_calculus:
            ops = {
                "<VARIABLE>": 0,
                "<ABSTRACTION>": 1,
                "<NUMBER>": 2,
                "<BOOLEAN>": 3,
                "<NIL>": 4,
                "<IF>": 5,
                "<CONS>": 6,
                "<MATCH>": 7,
                "<UNARYOPER>": 8,
                "<BINARYOPER>": 9,
                "<LET>": 10,
                "<LETREC>": 11
            }
        else:
            ops = {
                "<VAR>": 0,
                "<CONST>": 1,
                "<PLUS>": 2,
                "<MINUS>": 3,
                "<EQUAL>": 4,
                "<LE>": 5,
                "<GE>": 6,
                "<ASSIGN>": 7,
                "<IF>": 8,
                "<SEQ>": 9,
                "<FOR>": 10
            }
            

        progsjson = json.load(open(path))
        progs = [make_tree(prog, long_base_case=long_base_case, 
                           is_lambda_calculus=is_lambda_calculus) for prog in progsjson]
        if binarize:
            progs = [binarize_tree(prog) for prog in progs]

        prog_len = num_vars + num_ints + len(ops.keys())
        progs = [encode_tree(prog, num_vars, num_ints, ops, eos_token=input_eos_token) for prog in progs]
        
        
        self.for_data_pairs = list(zip(for_progs, lambda_progs))
            