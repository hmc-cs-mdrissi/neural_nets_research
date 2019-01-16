import torch
from torch.utils.data import Dataset
from tree_to_sequence.translating_trees import *
from math_expressions.translating_math_trees import *


import copy
import json
from random import shuffle
import pickle
import cv2
import numpy as np
import math
import random

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
                 eos_token=True,  input_as_seq=False, output_as_seq=True, one_hot=False, sort_by_length=False):
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
        
        # Sort dataset by length
        if sort_by_length:
            self.program_pairs.sort(key=lambda x: x[0].size())


    def __len__(self):
        return len(self.program_pairs)

    def __getitem__(self, index):
        return self.program_pairs[index]

import matplotlib.pyplot as plt
def display_normally(pic, title=None):
    if not title is None:
        plt.title(title)
    plt.imshow(np.repeat(np.int0(pic)[:,:,np.newaxis], 3, axis=2))
    plt.show()

    
def load_shuffled_data(data_path):
    with open(data_path, 'rb') as f:
        program_pairs = pickle.load(f)
    shuffle(program_pairs)
    return program_pairs
    
    
class MathExprDatasetBatched(Dataset):
    def __init__(self, 
                 program_pairs=None,
                 data_path=None, 
                 batch_size=128,
                 binarize_output=False,
                 eos_token=True, 
                 max_children_output=3,
                 num_samples=None,
                 normalize=False,
                 trim_factor=1,
                 left_align=True,
                 num_copies=1,
                 max_depth=None):
    
        assert(program_pairs or data_path)
        
        if data_path:
            with open(data_path, 'rb') as f:
                program_pairs = pickle.load(f)
        
        def sort_by_img_size(element):
            return -1 * element[0].shape[1]
        
        def sort_by_tree_size(element):
            return element[1].size()
        
        self.trim_factor = trim_factor
        self.left_align = left_align
        
        # Sort by width of expression
        program_pairs = sorted(program_pairs, key=sort_by_img_size)[:num_samples] 
        
        input_programs = [img for img, tree in program_pairs]
        output_programs = [tree for img, tree in program_pairs]
    
        if not max_depth is None:
            output_programs = [self.trim_depth(tree, max_depth) for tree in output_programs]
        
        # Format the input trees correctly
        input_programs = [self.process_img(img) for img in input_programs]
        
        if normalize:
            input_programs = [img/255.0 for img in input_programs]
        
        # Binarize output trees if specified
        if binarize_output:
            output_programs = map(binarize_tree, output_programs)
            
        # Add EOS tokens if specified
        if eos_token:
            output_programs = map(lambda prog: add_eos(prog, num_children=max_children_output), output_programs)
        
        # Turn strings into numbers in the output trees
        output_programs = [encode_math_program(prog, math_tokens_short) for prog in output_programs]

        self.program_pairs = []
        
        for _ in range(num_copies):
            # Chop into batches
            index = 0
            all_batches = []
            while index < len(input_programs):
                end_index = min(len(input_programs), index + batch_size)
                all_batches.append(self.batch_images((input_programs[index:end_index], output_programs[index:end_index])))
                index = end_index
            
            # Shuffle so you don't get all the small ones first
            shuffle(all_batches[1:]) #TODO: ADD HTIS BACK
            
            self.program_pairs = self.program_pairs + all_batches
        
    def trim_depth(self, tree, max_depth):
        if max_depth == 1:
            tree.children = []
            return tree
        
        tree.children = [self.trim_depth(child, max_depth - 1) for child in tree.children]
        return tree
    
    def batch_images(self, batch):
        num = len(batch[0])
        images = batch[0]
        width = 0
        height = 305 #current height of image
        padded = []
        for image in images:
            # Find the largest width
            if image.shape[3] > width:
                width = image.shape[3]
        
        blank = np.ones((num, 1, height, width), dtype=float)
        for i, image in enumerate(images):
            img_width = image.shape[3]
            if self.left_align:
                start_index = 0
            else:
                max_index = width - img_width
                start_index = random.randint(0, max_index)
                
            blank[i, :, :, start_index:start_index + img_width] = image
        batch_img = torch.FloatTensor(blank)
        batch_img = self.trim_by_factor(batch_img, self.trim_factor)
        return (batch_img, batch[1])
        
    
    def process_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
        return img   
    
    def trim_by_factor(self, img, factor):
        trim_amount = img.shape[3] % factor
        if trim_amount > 0:
            img = img[:, :, :, :-trim_amount]
        batch, channels, height, width = img.shape
        new_img = torch.nn.functional.interpolate(img, size=(height, width // factor), mode="bilinear")
        return torch.FloatTensor(new_img)
    
    def __len__(self):
        return len(self.program_pairs)

    def __getitem__(self, index):
        return self.program_pairs[index]

    
class ForLambdaDatasetLengthBatched(SyntacticProgramDataset):
    def __init__(self, path, batch_size, num_vars=10, num_ints=11, binarize_input=False, binarize_output=False, eos_token=True, 
                 input_as_seq=False, output_as_seq=True, one_hot=True, long_base_case=True, num_samples=None):
        progs_json = json.load(open(path))[:num_samples]
        for_progs = [make_tree_for(prog, long_base_case=long_base_case) for prog in progs_json]
        lambda_progs = [translate_from_for(copy.deepcopy(for_prog)) for for_prog in for_progs]

        max_children_output = 2 if binarize_output else max_children_lambda
        super().__init__(for_progs, lambda_progs, input_ops=for_ops, output_ops=lambda_ops,
                         max_children_output=max_children_output, num_vars=num_vars, 
                         num_ints=num_ints, binarize_input=binarize_input, binarize_output=binarize_output, 
                         eos_token=eos_token, input_as_seq=input_as_seq, output_as_seq=output_as_seq, one_hot=one_hot,
                         sort_by_length=True)
        
    
        index = 0
        all_batches = []
        while index < len(self.program_pairs):
            batch_items = []
            size = self.program_pairs[index][0].size()
            for item_index in range(index, min(len(self.program_pairs), index + batch_size)):
                if self.program_pairs[item_index][0].size() == size:
                    batch_items.append(self.program_pairs[item_index])
                    index += 1
                else:
                    break
            all_batches.append(batch_items)


        shuffle(all_batches)
        print("Num batches", len(all_batches))
        print("total length is", len(self.program_pairs))
        print("batch size is", batch_size)
        print("shortest", min([len(x) for x in all_batches]))
        print("longest", max([len(x) for x in all_batches]))
        print("shortest", min([len(x) for x in all_batches]))
        
        self.program_pairs = all_batches
        
    def __len__(self):
        return len(self.program_pairs)

    def __getitem__(self, index):
        return self.program_pairs[index]
        

class ForLambdaDataset(SyntacticProgramDataset):
    def __init__(self, path, num_vars=10, num_ints=11, binarize_input=False, binarize_output=False, eos_token=True, 
                 input_as_seq=False, output_as_seq=True, one_hot=True, long_base_case=True, num_samples=None, sort_by_length=False):
        progs_json = json.load(open(path))[:num_samples]
        for_progs = [make_tree_for(prog, long_base_case=long_base_case) for prog in progs_json]
        lambda_progs = [translate_from_for(copy.deepcopy(for_prog)) for for_prog in for_progs]

        max_children_output = 2 if binarize_output else max_children_lambda
        super().__init__(for_progs, lambda_progs, input_ops=for_ops, output_ops=lambda_ops,
                         max_children_output=max_children_output, num_vars=num_vars, 
                         num_ints=num_ints, binarize_input=binarize_input, binarize_output=binarize_output, 
                         eos_token=eos_token, input_as_seq=input_as_seq, output_as_seq=output_as_seq, one_hot=one_hot,
                         sort_by_length=sort_by_length)

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
