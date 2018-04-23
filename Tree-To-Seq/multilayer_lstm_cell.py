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

class MultilayerLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_layers, bias=True):
        super(MultilayerLSTMCell, self).__init__()
        self.lstm_layers = nn.ModuleList()
        
        if isinstance(hidden_sizes, int):
            temp = []
            
            for _ in range(num_layers):
                temp.append(hidden_sizes)
            
            hidden_sizes = temp
            
        hidden_sizes = [input_size] + hidden_sizes
        
        for i in range(num_layers):
            curr_lstm = nn.LSTMCell(hidden_sizes[i], hidden_sizes[i+1], bias=bias)
            self.lstm_layers.append(curr_lstm)
    
    def initialize_forget_bias(self, bias_value):
        for lstm_cell in self.lstm_layers:
            n = lstm_cell.bias_ih.size(0)
            start, end = n//4, n//2
            b1 = lstm_cell.bias_ih
            nn.init.constant(lstm_cell.bias_ih[start:end], bias_value)
            nn.init.constant(lstm_cell.bias_hh[start:end], bias_value)
    
    def forward(self, input, past_states):
        hiddens, cell_states = past_states
        result_hiddens, result_cell_states = [], []
        curr_input = input
        
        for lstm_cell, curr_hidden, curr_cell_state in zip(self.lstm_layers, hiddens, cell_states):
            curr_input, new_cell_state = lstm_cell(curr_input, (curr_hidden, curr_cell_state))
            result_hiddens.append(curr_input)
            result_cell_states.append(new_cell_state)
        
        return torch.stack(result_hiddens), torch.stack(result_cell_states)