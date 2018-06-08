import torch
import torch.nn as nn

from tree_to_sequence.tree_lstm import TreeLSTM
from tree_to_sequence.translating_trees import map_tree, tree_to_list

class TreeEncoder(nn.Module):
    """
    Takes in a tree where each node has a value vector and a list of children
    Produces a sequence encoding of the tree. valid_num_children is not needed
    if you choose to use a binary_lstm_cell.
    """
    def __init__(self, input_size, hidden_size, num_layers, valid_num_children=None,
                 attention=True, one_hot=False, embedding_size=256, dropout=False,
                 binary_tree_lstm_cell=False):
        super(TreeEncoder, self).__init__()

        self.lstm_list = nn.ModuleList()
        self.one_hot = one_hot
        self.dropout = False
        
        if dropout:
            self.dropout = nn.Dropout(p=dropout)

        if one_hot:
            if binary_tree_lstm_cell:
                self.lstm_list.append(BinaryTreeLSTM(input_size, hidden_size))
            else:
                self.lstm_list.append(TreeLSTM(input_size, hidden_size, valid_num_children))
        else:
            self.embedding = nn.Embedding(input_size, embedding_size)
            self.lstm_list.append(TreeLSTM(embedding_size, hidden_size, valid_num_children))
            

        # All TreeLSTMs have input of hidden_size except the first.
        for i in range(num_layers-1):
            if binary_tree_lstm_cell:
                self.lstm_list.append(BinaryTreeLSTM(input_size, hidden_size))
            else:
                self.lstm_list.append(TreeLSTM(input_size, hidden_size, valid_num_children))

        self.attention = attention

    def forward(self, tree):
        """
        Encodes nodes of a tree in the rows of a matrix.

        :param tree: a tree where each node has a value vector and a list of children
        :return a matrix where each row represents the encoded output of a single node and also
                the hidden/cell states of the root node.

        """
        if not self.one_hot:
            tree = map_tree(lambda node: self.embedding(node).squeeze(0), tree)
            
        if self.dropout:
            tree = map_tree(lambda node: self.dropout(node), tree)

        hiddens = []
        cell_states = []

        for lstm in self.lstm_list:
            tree, cell_state = lstm(tree)
            hiddens.append(tree.value)
            cell_states.append(cell_state)

        hiddens = torch.stack(hiddens)
        cell_states = torch.stack(cell_states)

        if self.attention:
            annotations = torch.stack(list(filter(lambda x: x is not None, tree_to_list(tree))))
            return annotations, hiddens, cell_states
        else:
            return hiddens, cell_states

    def initialize_forget_bias(self, bias_value):
        for lstm in self.lstm_list:
            lstm.initialize_forget_bias(bias_value)
