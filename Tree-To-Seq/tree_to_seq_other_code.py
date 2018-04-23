import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import random
import matplotlib.pyplot as plt

from neural_nets_library import training, visualize
from ANC import Controller

import json
from translating_trees import *
from for_prog_dataset import ForDataset
from functools import partial
from tree_lstm import *
from seq_encoder import *
from tree_to_seq_model import *
from tree_to_seq_attention_model import *
from tree_to_seq_attention_anc_model import *


num_vars = 10
num_ints = 11

for_ops = {
    "Var": 0,
    "Const": 1,
    "Plus": 2,
    "Minus": 3,
    "EqualFor": 4,
    "LeFor": 5,
    "GeFor": 6,
    "Assign": 7,
    "If": 8,
    "Seq": 9,
    "For": 10
}

for_ops = {"<" + k.upper() + ">": v for k,v in for_ops.items()}

lambda_ops = {
    "Var": 0,
    "Const": 1,
    "Plus": 2,
    "Minus": 3,
    "EqualFor": 4,
    "LeFor": 5,
    "GeFor": 6,
    "If": 7,
    "Let": 8,
    "Unit": 9,
    "Letrec": 10,
    "App": 11
}

lambda_ops = {"<" + k.upper() + ">": v for k,v in lambda_ops.items()}

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

input_eos_token = False
input_as_seq = False
use_embedding = True
eos_bonus = 1 if input_eos_token and input_as_seq else 0
long_base_case = True
binarize = True

is_lambda_calculus = False

for_anc_dset = TreeANCDataset("ANC/Easy-arbitraryForListWithOutput.json", is_lambda_calculus, binarize=binarize, input_eos_token=input_eos_token, 
                              use_embedding=use_embedding, long_base_case=long_base_case, 
                              input_as_seq=input_as_seq, cuda=True)

def reset_all_parameters_uniform(model, stdev):
    for param in model.parameters():
        nn.init.uniform(param, -stdev, stdev)

embedding_size = 30
hidden_size = 30
num_layers = 1
alignment_size = 50
align_type = 1
M, R = 10, 3
N = 11
encoder_input_size = num_vars + num_ints + len(for_ops.keys()) + eos_bonus

if input_as_seq:
    encoder = SeqEncoder(encoder_input_size, hidden_size, num_layers, attention=True, use_embedding=use_embedding)
else:
    encoder = TreeEncoder(encoder_input_size, hidden_size, num_layers, [1, 2], attention=True, use_embedding=use_embedding)

decoder = MultilayerLSTMCell(N + 3*R + hidden_size, hidden_size, num_layers)
program_model = Tree_to_Sequence_Attention_ANC_Model(encoder, decoder, hidden_size, embedding_size, M, R, 
                                                     alignment_size=alignment_size, align_type=align_type)
    
reset_all_parameters_uniform(program_model, 0.1)
encoder.initialize_forget_bias(3)
decoder.initialize_forget_bias(3)


program_modelprogram  = program_model.cuda()


optimizer = torch.optim.Adam(program_model.parameters(), lr=0.5)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=100, factor=0.8)

for prog, target in for_anc_dset:
    controller = program_model(prog)
    controller.cuda()
    loss = program_model.compute_loss(controller, target)
    loss.backward()
    
    for name, param in program_model.named_parameters():
        print(name)
        print(param.grad)
    optimizer.zero_grad()
    break

import importlib
importlib.reload(training)

_ = \
    training.train_model_tree_to_anc(program_model, for_anc_dset, optimizer, 
                                     lr_scheduler=lr_scheduler, 
                                     num_epochs=10, batch_size=1,
                                     cuda=True,
                                     plateau_lr=True,
                                     print_every=100)

program_model.encoder.embedding.weight.grad