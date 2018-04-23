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

class Tree_to_Sequence_Attention_ANC_Model(Tree_to_Sequence_Attention_Model):
    def __init__(self, encoder, decoder, hidden_size, embedding_size, M, R,
                 alignment_size=50, align_type=1, N=11, t_max=10):
        # The 1 is for nclasses which is not used in this model.
        super(Tree_to_Sequence_Attention_ANC_Model, self).__init__(encoder, decoder, hidden_size, 1, embedding_size,
                                                                   alignment_size=alignment_size, align_type=align_type)
        # the initial registers all have value 0 with probability 1
        prob_dist = torch.zeros(R, M)
        prob_dist[:, 0] = 1
        
        self.register_buffer('initial_registers', prob_dist)
        
        self.M = M
        self.R = R
        self.N = N
        self.t_max = t_max
        
        self.initial_word_input = nn.Parameter(torch.Tensor(1, N + 3*R))
        self.output_log_odds = nn.Linear(hidden_size, N + 3*R)
        
    """
        input: The output of the encoder for the tree should have be a triple. The first 
               part of the triple should be the annotations and have dimensions, 
               number_of_nodes x hidden_size. The second triple of the pair should be the hidden 
               representations of the root and should have dimensions, num_layers x hidden_size.
               The third part should correspond to the cell states of the root and should
               have dimensions, num_layers x hidden_size.
    """
    def forward(self, input):
        annotations, decoder_hiddens, decoder_cell_states = self.encoder(input)
        # align_size: 0 number_of_nodes x alignment_size or align_size: 1-2 bengio number_of_nodes x hidden_size
        if self.align_type <= 1:
            attention_hidden_values = self.attention_hidden(annotations)
        else:
            attention_hidden_values = annotations
        
        decoder_hiddens = decoder_hiddens.unsqueeze(1) # num_layers x 1 x hidden_size
        decoder_cell_states = decoder_cell_states.unsqueeze(1) # num_layers x 1 x hidden_size

        word_input = self.initial_word_input # 1 x N + 3*R
        et = Variable(self.et)
        
        output_words = []

        for i in range(self.M):
            decoder_input = torch.cat((word_input, et), dim=1) # 1 x N + 3*R + hidden_size
            decoder_hiddens, decoder_cell_states = self.decoder(decoder_input, (decoder_hiddens, decoder_cell_states))
            decoder_hidden = decoder_hiddens[-1]
            
            attention_logits = self.attention_logits(attention_hidden_values, decoder_hidden)
            attention_probs = self.softmax(attention_logits) # number_of_nodes x 1
            context_vec = (attention_probs * annotations).sum(0).unsqueeze(0) # 1 x hidden_size
            et = self.tanh(self.attention_presoftmax(torch.cat((decoder_hidden, context_vec), dim=1)))
            word_input = self.output_log_odds(et)
            
            output_words.append(word_input)
            
        controller_params = torch.stack(output_words, dim=2).squeeze(0) # N + 3*R x M
        instruction = controller_params[0:N]
        first_arg = controller_params[N:N+R]
        second_arg = controller_params[N+R:N+2*R]
        output = controller_params[N+2*R:N + 3*R]
        controller = Controller.Controller(first_arg=first_arg, second_arg=second_arg, output=output, 
                                           instruction=instruction, 
                                           initial_registers=Variable(self.initial_registers),
                                           multiplier = 1, correctness_weight=1, halting_weight=1, 
                                           confidence_weight=0, efficiency_weight=0, t_max=self.t_max)
        
        return controller
        
    """
        controller: The controller for an ANC.
        target: The target should be a list of triples, where the first element of any triple is
                the input matrix, the second element is the output matrix corresponding to the expected
                output based on the input and the third element is a mask that specifies the area
                of memory where the output is.
    """
    def compute_loss(self, controller, target):
        loss = 0
        input_memories = target[0]
        output_memories = target[1]
        output_masks = target[2]
        
        for i in range(len(input_memories)):
            loss += controller.forward_train(input_memories[i], (output_memories[i], output_masks[i]))
            
        return loss