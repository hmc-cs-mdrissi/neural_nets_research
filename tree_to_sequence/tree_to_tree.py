import torch
import torch.nn as nn
from torch.autograd import Variable
from tree_to_sequence.translating_trees import Node
from tree_to_sequence.translating_trees import print_tree
from tree_to_sequence.translating_trees import pretty_print_tree


class TreeToTree(nn.Module):
    def __init__(self, encoder, decoder, hidden_size, embedding_size,
                 alignment_size=50, align_type=1):
        super(TreeToTree, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.i = 0

    def forward_train(self, input_tree, target_tree):
        loss = 0.0

        annotations, decoder_hiddens, decoder_cell_states = self.encoder(input_tree)
        loss = self.decoder.forward_train(decoder_hiddens, decoder_cell_states, target_tree, annotations)
        return loss
        
    def forward_prediction(self, input_tree, target_tree):
        annotations, decoder_hiddens, decoder_cell_states = self.encoder(input_tree)
        tree = self.decoder.forward_prediction(decoder_hiddens, decoder_cell_states, annotations)
        return tree
    
    def print_example(self, input_tree, target_tree):
        print("DESIRED")
        pretty_print_tree(target_tree)
        print("WE GOT!")
        pretty_print_tree(self.forward_prediction(input_tree, target_tree))


    def extract_best(self, tree): 
        tree.value = tree.value[2]
        for child in tree.children:
            self.extract_best(child)
         
        

