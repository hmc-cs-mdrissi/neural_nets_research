import torch
import torch.nn as nn

from tree_to_sequence.tree_lstm import TreeCell
from tree_to_sequence.translating_trees import map_tree, tree_to_list

class TreeEncoderBatch(nn.Module):
    """
    Takes in a tree where each node has a value vector and a list of children
    Produces a sequence encoding of the tree. 
    """
    def __init__(self, input_size, hidden_size):
        super(TreeEncoderBatch, self).__init__()
        
        self.tree_lstm = TreeCell(input_size, hidden_size, num_children=2)            
        self.register_buffer('zero_buffer', torch.zeros(1, hidden_size)) 
        
        # Will be set later by the tree_to_tree file.
        self.embedding = None


    def forward(self, fold, node):
        """
        Encodes nodes of a tree in the rows of a matrix.

        :param fold: torchfold object
        :param tree: a tree where each node has a value vector and a list of children
        :return a matrix where each row represents the encoded output of a single node and also
                the hidden/cell states of the root node.

        """
        value = self.embedding(node.value)
        
        if value is None:
            return fold.add('encode_none_node').split(3)

        # List of tuples: (node, cell state)
        children = []

        for child in node.children:
            encoded = self.forward(fold, child)
            children += list(encoded)

        while len(children) < 6:
            children += fold.add('encode_none_node').split(3)

        return  fold.add('encode_node_with_children', value, *children).split(3)
    
    def encode_none_node(self):
        """
        Returns dummy zero vectors for the children of a node which doesn't have a child in that position

        :return annotations, hidden_state, cell_state
        """
        return self.zero_buffer.unsqueeze(1), self.zero_buffer, self.zero_buffer
    
    # TODO: Later make this stackable
    def encode_node_with_children(self, value, leftA, leftH, leftC, rightA, rightH, rightC):
        """
        Returns the encoding of a node with children.
        
        :param value: node's value
        :param leftA: left child's annotations
        :param leftH: left child's hidden state
        :param leftC: left child's cell state
        :param rightA: left child's annotations
        :param rightH: right child's hidden state
        :param rightC: right child's cell state
        
        :return annotations, hidden state, cell state
        """
        newH, newC = self.tree_lstm(value, [leftH, rightH], [leftC, rightC])
        newA = newH.unsqueeze(1)
        newA = torch.cat([newA, leftA.float(), rightA.float()])
        return newA, newH, newC
        

    def initialize_forget_bias(self, bias_value):
        """
        Initialize the forget bias to a certain value. Primary purpose is that initializing
        with a largish value (like 3) tends to help convergence by preventing the model
        from forgetting too much early on.
        
        :param bias_value: value to set the bias to
        """
        self.tree_lstm.initialize_forget_bias(bias_value)
