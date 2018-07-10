import torch
from torch.utils.data import Dataset
from tree_to_sequence.translating_trees import *

import copy
import json

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
max_children_for = 5

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
            "<APP>": 11,
        }
max_children_lambda = 4

lambda_calculus_ops = {
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
max_children_lambda_calculus = 3

coffee_ops = {
    "<VAR>": 0,
    "<CONST>": 1,
    "<PLUS>": 2,
    "<TIMES>": 3,
    "<EQUAL>": 4,
    "<ASSIGN>": 5,
    "<IF>": 6,
    "<IFSIMPLE>": 7,
    "<SIMPLEIF>": 8,
    "<IFELSE>": 9,
    "<IFTHENELSE>": 10,
    "<IFCOMPLEX>": 11,
    "<SIMPLECS>": 12,
    "<COMPLEXCS>": 13,
    "<EXPR>": 14,
    "<SHORTSTATEMENTCS>": 15,
    "<WHILE>": 16,
    "<WHILESIMPLE>": 17,
    "<SIMPLEWHILE>": 18,
    "<WHILECOMPLEX>": 19,
    "<SIMPLESTATEMENT>": 20,
}
max_children_coffee = 3

javascript_ops = {
    '<IFSTATEMENT>': 0,
    '<VARIABLEDECLARATOR>': 1, 
    '<ASSIGNMENTEXPRESSION>': 2, 
    '<LITERAL>': 3, 
    '+': 4,
    'results2': 5, 
    'call': 6, 
    '<IDENTIFIER>': 7, 
    '<PROGRAM>': 8, 
    '<RETURNSTATEMENT>': 9, 
    '<VARIABLEDECLARATION>': 10, 
    '<THISEXPRESSION>': 11, 
    'results3': 12, 
    '<FUNCTIONEXPRESSION>': 13, 
    '===': 14, 
    '*': 15, 
    '<ARRAYEXPRESSION>': 16, 
    'results': 17, 
    '<BLOCKSTATEMENT>': 18, 
    '<EXPRESSIONSTATEMENT>': 19, 
    '=': 20, 
    '<MEMBEREXPRESSION>': 21, 
    '<CALLEXPRESSION>': 22, 
    'results1': 23, 
    '<WHILESTATEMENT>': 24, 
    'push': 25,    
}



class SyntacticProgramDataset(Dataset):
    def __init__(self, input_programs, output_programs, input_ops=None, output_ops=None, 
                         max_children_output=None, num_vars=10, num_ints=11, binarize_input=False, binarize_output=False, 
                 eos_token=True,  input_as_seq=False, output_as_seq=True, one_hot=False):
        if eos_token and not output_as_seq and max_children_output is None:
            raise ValueError("When the output is a tree and you want end of tree tokens, it is"
                             " necessary that max_children_output is not None.")
        
        if binarize_output and not output_as_seq and not eos_token:
            raise ValueError("When the output is a binarized tree, you must have end of tree "
                             "tokens.")
        
        if binarize_input:
            input_programs = map(binarize_tree, input_programs)
            input_programs = map(clean_binarized_tree, input_programs)
        
        if binarize_output:
            output_programs = map(binarize_tree, output_programs)
        
        if input_as_seq:
            input_programs = map(lambda ls: filter(lambda x: x is not None, ls), 
                                 map(tree_to_list, input_programs))
            
        if output_as_seq:
            output_programs = map(lambda ls: filter(lambda x: x is not None, ls),
                                  map(tree_to_list, output_programs))
        
        if eos_token:
            output_programs = map(lambda prog: add_eos(prog, num_children=max_children_output), 
                                  output_programs)
            
        input_programs = [encode_program(prog, num_vars, num_ints, input_ops, eos_token=eos_token, 
                                         one_hot=one_hot) for prog in input_programs]
        output_programs = [encode_program(prog, num_vars, num_ints, output_ops, eos_token=eos_token) for prog in output_programs]
        self.program_pairs = list(zip(input_programs, output_programs))

    def __len__(self):
        return len(self.program_pairs)

    def __getitem__(self, index):
        return self.program_pairs[index]
    
class ForLambdaDataset(SyntacticProgramDataset):
    def __init__(self, path, num_vars=10, num_ints=11, binarize_input=False, binarize_output=False, eos_token=True, 
                 input_as_seq=False, output_as_seq=True, one_hot=True, long_base_case=True):
        progs_json = json.load(open(path))
        for_progs = [make_tree_for(prog, long_base_case=long_base_case) for prog in progs_json]
        lambda_progs = [translate_from_for(copy.deepcopy(for_prog)) for for_prog in for_progs]

        max_children_output = 2 if binarize_output else max_children_lambda
        super().__init__(for_progs, lambda_progs, input_ops=for_ops, output_ops=lambda_ops,
                         max_children_output=max_children_output, num_vars=num_vars, 
                         num_ints=num_ints, binarize_input=binarize_input, binarize_output=binarize_output, 
                         eos_token=eos_token, input_as_seq=input_as_seq, output_as_seq=output_as_seq, one_hot=one_hot)

class JsCoffeeDataset(SyntacticProgramDataset):
    def __init__(self, coffeescript_path, javascript_path, num_vars=10, num_ints=11, binarize_input=False, binarize_output=False, eos_token=True, 
                 input_as_seq=False, output_as_seq=True, one_hot=True, long_base_case=True):
        coffeescript_progs = [make_tree_coffeescript(prog, long_base_case=long_base_case) for prog in json.load(open(coffeescript_path))]
        javascript_progs = [make_tree_javascript(prog, long_base_case=long_base_case) for prog in json.load(open(javascript_path))]

        max_children_output = 2 if binarize_output else max_children_coffee
        super().__init__(javascript_progs, coffeescript_progs, input_ops=javascript_ops, output_ops=coffee_ops,
                         max_children_output=max_children_output, num_vars=num_vars, 
                         num_ints=num_ints, binarize_input=binarize_input, binarize_output=binarize_output, 
                         eos_token=eos_token, input_as_seq=input_as_seq, output_as_seq=output_as_seq, one_hot=one_hot)         
        
class SemanticProgramDataset(Dataset):
    def __init__(self, is_lambda_calculus, num_vars=10, num_ints=11, binarize=False, 
                 input_as_seq=False, one_hot=False, long_base_case=True, cuda=True):
        if is_lambda_calculus:
            self.ops = lambda_calculus_ops
        else:
            self.ops = for_ops
        
        self.num_vars = num_vars
        self.num_ints = num_ints
        self.binarize = binarize
        self.is_lambda_calculus = is_lambda_calculus
        self.one_hot = one_hot
        self.input_as_seq = input_as_seq
        self.cuda = cuda
        
    def construct_input_program(self, program_json):
        if self.is_lambda_calculus:
            program = make_tree_lambda_calculus(program_json)
        else:
            program = make_tree_for(program_json)

        if self.binarize:
            program = binarize_tree(program)

        token_size = self.num_vars + self.num_ints + len(self.tokens.keys())
        
        if self.input_as_seq:
            program = filter(lambda x: x is not None, tree_to_list(program))
                    
        program = encode_program(program, self.num_vars, self.num_ints, self.ops, 
                                 one_hot=self.one_hot)
        
        if self.cuda:
            program = program.cuda()
        
        return program
        
class TreeANCDataset(SemanticProgramDataset):
    def __init__(self, path, is_lambda_calculus, num_vars=10, num_ints=11, binarize=False,
                 input_as_seq=False, one_hot=False, long_base_case=True, cuda=True):
        super().__init__(is_lambda_calculus, num_vars=num_vars, num_ints=num_ints, 
                         binarize=binarize, input_as_seq=input_as_seq, one_hot=one_hot, 
                         long_base_case=long_base_case, cuda=cuda)

        progs_json = json.load(open(path))
        self.progs = [self.convert_to_quadruple(prog_input_output) for prog_input_output in 
                      progs_json]

    def convert_to_quadruple(self, prog_input_output):
        program = construct_input_program(prog_input_output[0])

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

            input_matrices.append(input_matrix)
            output_matrices.append(output_matrix)
            masks.append(mask)

        return program, (input_matrices, output_matrices, masks)

    def __len__(self):
        return len(self.progs)

    def __getitem__(self, index):
        return self.progs[index]

class TreeNTMDataset(SemanticProgramDataset):
    def __init__(self, path, is_lambda_calculus, thinking_time, repeats=2, num_vars=10, num_ints=11, 
                 binarize=False, input_as_seq=False, one_hot=False, long_base_case=True, cuda=True):
        super().__init__(is_lambda_calculus, num_vars=num_vars, num_ints=num_ints, 
                         binarize=binarize, input_as_seq=input_as_seq, one_hot=one_hot, 
                         long_base_case=long_base_case, cuda=cuda)

        progs_json = json.load(open(path))
        self.progs = [self.convert_to_triple(prog_input_output, thinking_time, repeats) for 
                      prog_input_output in progs_json]

    def convert_to_triple(self, prog_input_output, thinking_time, repeats):
        program = construct_input_program(prog_input_output[0])
        inputs_outputs = torch.FloatTensor(prog_input_output[1]*repeats)
        inputs = inputs_outputs[:,0].unsqueeze(1)
        outputs = inputs_outputs[:,1]
        inputs = torch.cat((inputs, torch.zeros((inputs.size(0), thinking_time))), 1) 
        return prog_tree, inputs, outputs
            
    def __len__(self):
        return len(self.progs)

    def __getitem__(self, index):
        return self.progs[index]

class IdentityTreeToTreeDataset(SyntacticProgramDataset):
    def __init__(self, programs, ops, max_children_output=None, num_vars=10, num_ints=11,
                 binarize=False, eos_token=True,  input_as_seq=False, output_as_seq=True, 
                 one_hot=False):
        super().__init__(programs, programs, input_ops=ops, output_ops=ops,
                         max_children_output=max_children_output, num_vars=num_vars, 
                         num_ints=num_ints, binarize=binarize, eos_token=eos_token,  
                         input_as_seq=input_as_seq, output_as_seq=output_as_seq, one_hot=one_hot)
        
class Const5(IdentityTreeToTreeDataset):
    def __init__(self, num_vars=10, num_ints=11, binarize=False, eos_token=True, input_as_seq=False, 
                 output_as_seq=True, one_hot=False, long_base_case=True):
        progs_json = json.loads('[{"tag": "Const", "contents": 5}]')*1000
        programs = [make_tree_lambda(prog, long_base_case=long_base_case) for prog in progs_json]
        max_children_output = 2 if binarize else max_children_lambda
        super().__init__(programs, lambda_ops, max_children_output=max_children_output,
                         num_vars=num_vars, num_ints=num_ints, binarize=binarize, 
                         eos_token=eos_token,  input_as_seq=input_as_seq,
                         output_as_seq=output_as_seq, one_hot=one_hot)
