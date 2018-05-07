import torch
import torch.nn as nn
from torch.autograd import Variable
from tree_to_sequence.translating_trees import Node

class TreeToTree(nn.Module):
    def __init__(self, encoder, decoder, hidden_size, embedding_size,
                 alignment_size=50, align_type=1):
        super(TreeToTree, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder

    def forward_train(self, input_tree, target_tree):
        print("TRAIN!")
        loss = 0.0
        prediction = self.forward_prediction(input_tree)
        matches = self.count_matches(prediction, target_tree)
        loss = matches * 1.0 / target_tree.size()

        return loss

    """
        This is just an alias for point_wise_prediction, so that training code that assumes the presence
        of a forward_train and forward_prediction works.
    """
    def forward_prediction(self, prog_input):
        return self.point_wise_prediction(prog_input)

    def point_wise_prediction(self, tree_input):
        annotations, decoder_hiddens, decoder_cell_states = self.encoder(tree_input)
        tree = self.decoder(decoder_hiddens, decoder_cell_states, annotations)
        best_tree = self.extract_best(tree)

        return tree

    def extract_best(self, tree): 
        tree.value = tree.value[2]
        for child in tree.children:
            self.extract_best(child)
    
    def count_matches(self, prediction, target):
        matches = 0
        if int(prediction.value) == int(target.value):
            matches += 1
        for i in range(min(len(target.children), len(prediction.children))):
            matches += self.count_matches(prediction.children[i], target.children[i])
        return matches
         
        

