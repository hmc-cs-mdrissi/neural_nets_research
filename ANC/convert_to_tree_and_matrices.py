import torch
from torch.autograd import Variable



class TreeANCDataset(Dataset):
    def __init__(self, path, is_lambda_calculus, num_vars = 10, num_ints = 11, binarize = False, 
                 input_eos_token=True, input_as_seq=False, use_embedding=False,
                 long_base_case=True):
        if is_lambda_calculus:
            self.ops = {
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
            self.ops = {
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
        progsjson = json.load(open(path))
        self.progs = [self.convert_to_quadruple(progsjson, is_lambda_calculus, binarize) for prog_input_output in progsjson]
       



    def convert_to_quadruple(self, prog_input_output, is_lambda_calculus, binarize):
        #the prog_tree code is a placeholder depending on how different make_tree for the two
        #end up being and what we call them
        if is_lambda_calculus:
            prog_tree = make_tree(prog_input_output[0])
        else:
            prog_tree = make_tree(prog_input_output[0])
        if binarize:
            prog_tree = binarize_tree(prog_tree)
        prog_tree = encode_tree(prog_tree, self.num_vars, self.num_ints, self.ops, eos_token=input_eos_token)

        input_matrix = torch.zeros(10, 10)
        input_matrix[0][int(prog_input_output[1][0])] = 1
        input_matrix[1:9] = 0.1
        output_matrix = torch.zeros(10, 10)
        input_matrix[0][int(prog_input_output[1][1])] = 1
        input_matrix[1:9] = 0.1
        mask = torch.zeros(10, 10)
        mask[0] = 1
        return prog_tree, input_matrix, output_matrix, mask

    def __len__(self):
        return len(self.progs)

    def __getitem__(self, index):
        return self.progs[index]
            