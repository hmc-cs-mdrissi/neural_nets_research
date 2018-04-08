import json
from torch.utils.data import Dataset
from translating_trees import *

class ForDataset(Dataset):
    def __init__(self, path, num_vars = 10, num_ints = 11, binarize = False, 
                 input_eos_token=True, input_as_seq=False, use_embedding=False):
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

        for_progs = [make_tree(prog) for prog in progsjson]
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
