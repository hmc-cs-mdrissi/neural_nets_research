import json
from torch.utils.data import Dataset
from translating_trees import *

class ForDataset(Dataset):
    def __init__(self, path, num_vars = 10, num_ints = 11, binarize = False):
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

        for_progs = [encode_tree(prog, num_vars, num_ints, for_ops) for prog in for_progs]
        lambda_progs = [encode_tree(prog, num_vars, num_ints, lambda_ops,  one_hot=False) for prog in lambda_progs]
        alphabet_size = num_vars + num_ints + len(lambda_ops.keys())
        
        self.for_data_pairs = [(for_prog, Variable(torch.LongTensor(tree_to_list(lambda_prog) + [alphabet_size + 1]))) for for_prog, lambda_prog in zip(for_progs, lambda_progs)]

    def __len__(self):
        return len(self.for_data_pairs)

    def __getitem__(self, index):
        return self.for_data_pairs[index]