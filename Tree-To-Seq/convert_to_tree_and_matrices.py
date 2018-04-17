import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from translating_trees import *

import json

class TreeANCDataset(Dataset):
    def __init__(self, path, is_lambda_calculus, num_vars = 10, num_ints = 11, binarize = False, 
                 input_eos_token=True, input_as_seq=False, use_embedding=False,
                 long_base_case=True):
        if is_lambda_calculus:
            self.tokens = {
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
                "<LETREC>": 11,
                "<TRUE>": 12,
                "<FALSE>": 13,
                "<TINT>": 14,
                "<TBOOL>": 15,
                "<TINTLIST>": 16,
                "<TFUN>": 17,
                "<ARGUMENT>": 18,
                "<NEG>": 19,
                "<NOT>": 20,
                "<PLUS>": 21,
                "<MINUS>": 22,
                "<TIMES>": 23,
                "<DIVIDE>": 24,
                "<AND>": 25,
                "<OR>": 26,
                "<EQUAL>": 27,
                "<LESS>": 28,
                "<APPLICATION>": 29,
                "<HEAD>": 30,
                "<TAIL>": 31
            }
        else:
            self.tokens = {
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
            
        self.num_vars = num_vars
        self.num_ints = num_ints
        self.input_eos_token = input_eos_token
        self.binarize = binarize
        self.is_lambda_calculus = is_lambda_calculus

        progsjson = json.load(open(path))
        self.progs = [self.convert_to_quadruple(prog_input_output) for prog_input_output in progsjson]
       
    def convert_to_quadruple(self, prog_input_output):
        #the prog_tree code is a placeholder depending on how different make_tree for the two
        #end up being and what we call them
        prog_tree = make_tree(prog_input_output[0], is_lambda_calculus=self.is_lambda_calculus)
        if self.binarize:
            prog_tree = binarize_tree(prog_tree)
        prog_tree = encode_tree(prog_tree, self.num_vars, self.num_ints, self.tokens, eos_token=self.input_eos_token)

        input_matrices = []
        output_matrices = []
        masks = []

        for input, output in prog_input_output[1]:
            input_matrix = torch.zeros(10, 10)
            input_matrix[0][input % 10] = 1
            input_matrix[1:9] = 0.1

            output_matrix = torch.zeros(10, 10)
            output_matrix[0][output % 10] = 1
            output_matrix[1:9] = 0.1

            mask = torch.zeros(10, 10)
            mask[0] = 1

            input_matrices.append(input_matrix)
            output_matrices.append(output_matrix)
            masks.append(mask)

        return prog_tree, input_matrices, output_matrices, masks

    def __len__(self):
        return len(self.progs)

    def __getitem__(self, index):
        return self.progs[index]
            