import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from translating_trees import *

import json

class ForLambdaDataset(Dataset):
    def __init__(self, path, num_vars = 10, num_ints = 11, binarize = False,
                 input_eos_token=True, input_as_seq=False, use_embedding=False,
                 long_base_case=True):
        for_ops = {
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

        lambda_ops = {
            "<VAR>": 0,
            "<CONST>": 1,
            "<PLUS>": 2,
            "<MINUS>": 3,
            "<EQUAL>": 4,
            "<LE>": 5,
            "<GE>": 6,
            "<IF>": 7,
            "<LET>": 8,
            "<UNIT>": 9,
            "<LETREC>": 10,
            "<APP>": 11
        }

        progsjson = json.load(open(path))

        for_progs = [make_tree(prog, long_base_case=long_base_case) for prog in progsjson]
        lambda_progs = [translate_from_for(for_prog) for for_prog in for_progs]

        if binarize:
            for_progs = [binarize_tree(prog) for prog in for_progs]
            lambda_progs = [binarize_tree(prog) for prog in lambda_progs]

        for_size = num_vars + num_ints + len(for_ops.keys())
        lambda_size = num_vars + num_ints + len(lambda_ops.keys())

        if use_embedding:
            for_progs = [encode_tree(prog, num_vars, num_ints, for_ops, one_hot=False) for prog in for_progs]
            if input_as_seq:
                if input_eos_token:
                    for_progs = [Variable(torch.LongTensor(tree_to_list(prog) + [for_size])) for prog in for_progs]
                else:
                    for_progs = [Variable(torch.LongTensor(tree_to_list(prog))) for prog in for_progs]
            else:
                for_progs = [map_tree(lambda val: Variable(torch.LongTensor([val])), prog) for prog in for_progs]
        else:
            for_progs = [encode_tree(prog, num_vars, num_ints, for_ops, eos_token=input_eos_token) for prog in for_progs]
            if input_as_seq:
                if input_eos_token:
                    for_progs = [torch.stack(tree_to_list(prog) + [make_one_hot(for_size+1, for_size)]) for prog in for_progs]
                else:
                    for_progs =  [torch.stack(tree_to_list(prog)) for prog in for_progs]

        lambda_progs = [Variable(torch.LongTensor(tree_to_list(encode_tree(prog, num_vars, num_ints, lambda_ops,  one_hot=False)) + [lambda_size+1])) for prog in lambda_progs]

        self.for_data_pairs = list(zip(for_progs, lambda_progs))

    def __len__(self):
        return len(self.for_data_pairs)

    def __getitem__(self, index):
        return self.for_data_pairs[index]

class TreeANCDataset(Dataset):
    def __init__(self, path, is_lambda_calculus, num_vars = 10, num_ints = 11, binarize = False,
                 input_eos_token=True, input_as_seq=False, use_embedding=False,
                 long_base_case=True, cuda=True):
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
        self.use_embedding = use_embedding
        self.input_as_seq = input_as_seq
        self.cuda = cuda

        progsjson = json.load(open(path))
        self.progs = [self.convert_to_quadruple(prog_input_output) for prog_input_output in progsjson]

    def convert_to_quadruple(self, prog_input_output):
        #the prog_tree code is a placeholder depending on how different make_tree for the two
        #end up being and what we call them
        prog_tree = make_tree(prog_input_output[0], is_lambda_calculus=self.is_lambda_calculus)

        if self.binarize:
            prog_tree = binarize_tree(prog_tree)

        token_size = self.num_vars + self.num_ints + len(self.tokens.keys())

        if self.use_embedding:
            prog_tree = encode_tree(prog_tree, self.num_vars, self.num_ints, self.tokens, one_hot=False)
            if self.input_as_seq:
                if self.input_eos_token:
                    prog_tree = Variable(torch.LongTensor(tree_to_list(prog_tree) + [token_size]))
                else:
                    prog_tree = Variable(torch.LongTensor(tree_to_list(prog_tree)))
            else:
                prog_tree = map_tree(lambda val: Variable(torch.LongTensor([val])), prog_tree)
        else:
            prog_tree = encode_tree(prog_tree, self.num_vars, self.num_ints, self.tokens, eos_token=input_eos_token)
            if self.input_as_seq:
                if self.input_eos_token:
                    prog_tree = torch.stack(tree_to_list(prog_tree) + [make_one_hot(token_size+1, token_size)])
                else:
                    prog_tree = torch.stack(tree_to_list(prog_tree))

        if self.cuda:
            prog_tree = prog_tree.cuda()

        input_matrices = []
        output_matrices = []
        masks = []

        for input, output in prog_input_output[1]:
            input_matrix = torch.zeros(self.num_ints, self.num_ints)
            input_matrix[0][input % self.num_ints] = 1
            input_matrix[1:self.num_ints] = 0.1

            output_matrix = torch.zeros(self.num_ints, self.num_ints)
            output_matrix[0][output % self.num_ints] = 1
            output_matrix[1:self.num_ints] = 0.1

            mask = torch.zeros(self.num_ints, self.num_ints)
            mask[0] = 1

            if self.cuda:
                input_matrix, output_matrix, mask = input_matrix.cuda(), output_matrix.cuda(), mask.cuda()

            input_matrices.append(input_matrix)
            output_matrices.append(output_matrix)
            masks.append(mask)

        return prog_tree, (input_matrices, output_matrices, masks)

    def __len__(self):
        return len(self.progs)

    def __getitem__(self, index):
        return self.progs[index]

