import torch
import torch.nn as nn

from tree_to_sequence.translating_trees import Node

class TreeCell(nn.Module):
    """
    LSTM Cell which takes in arbitrary numbers of hidden and cell states (one per child).
    """
    def __init__(self, input_size, hidden_size, num_children):
        """
        Initialize the LSTM cell.

        :param input_size: length of input vector
        :param hidden_size: length of hidden vector (and cell state)
        :param num_children: number of children = number of hidden/cell states passed in
        """
        super(TreeCell, self).__init__()

        # Gates = input, output, memory + one forget gate per child
        numGates = 3 + num_children

        self.gates_value = nn.ModuleList()
        self.gates_children = nn.ModuleList()
        for _ in range(numGates):
            # One linear layer to handle the value of the node
            value_linear = nn.Linear(input_size, hidden_size, bias=True)
            children_linear = nn.ModuleList()
            # One per child of the node
            for _ in range(num_children):
                children_linear.append(nn.Linear(hidden_size, hidden_size, bias=False))
            self.gates_value.append(value_linear)
            self.gates_children.append(children_linear)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden_states, cell_states):
        """
        Calculate a new hidden state and a new cell state from the LSTM gates

        :param hidden_states: A list of num_children hidden states.
        :param cell_states: A list of num_children cell states.
        :return A tuple containing (new hidden state, new cell state)
        """

        data_sums = []

        for i in range(len(self.gates_value)):
            data_sum = self.gates_value[i](input)
            for j in range(len(hidden_states)):
                data_sum += self.gates_children[i][j](hidden_states[j])
            data_sums.append(data_sum)

        # First gate is the input gate
        input_val = self.sigmoid(data_sums[0])
        # Next output gate
        o = self.sigmoid(data_sums[1])
        # Next memory gate
        m = self.tanh(data_sums[2])
        # All the rest are forget gates
        forget_data = 0
        for i in range(len(cell_states)):
            forget_data += self.sigmoid(data_sums[3 + i]) * cell_states[i]

        # Put it all together!
        new_state = input_val * m + forget_data
        new_hidden = o * self.tanh(new_state)

        return new_hidden, new_state

    def initialize_forget_bias(self, bias_value):
        for i in range(3, len(self.gates_value)):
            nn.init.constant_(self.gates_value[i].bias, bias_value)

class TreeLSTM(nn.Module):
    '''
    TreeLSTM

    Takes in a tree where each node has a value and a list of children.
    Produces a tree of the same size where the value of each node is now encoded.
    '''

    def __init__(self, input_size, hidden_size, valid_num_children):
        """
        Initialize tree cells we'll need later.
        """
        super(TreeLSTM, self).__init__()

        self.valid_num_children = [0] + valid_num_children
        self.lstm_list = nn.ModuleList()

        for size in self.valid_num_children:
            self.lstm_list.append(TreeCell(input_size, hidden_size, size))
            
        self.register_buffer('zero_buffer', torch.zeros(hidden_size))

    def forward(self, node):
        """
        Creates a tree where each node's value is the encoded version of the original value.

        :param tree: a tree where each node has a value vector and a list of children
        :return a tuple - (root of encoded tree, cell state)
        """
        value = node.value
        
        if value is None:
            return (Node(None), self.zero_buffer)
        
        # List of tuples: (node, cell state)
        children = []

        # Recursively encode children
        for child in node.children:
            encoded_child = self.forward(child)
            children.append(encoded_child)

        # Extract the TreeCell inputs
        inputH = [vec[0].value for vec in children]
        inputC = [vec[1] for vec in children]

        for i, hidden in enumerate(inputH):
            if hidden is None:
                inputH[i] = self.zero_buffer
        
        found = False

        # Feed the inputs into the TreeCell with the appropriate number of children.
        for i in range(len(self.valid_num_children)):
            if self.valid_num_children[i] == len(children):
                newH, newC = self.lstm_list[i](value, inputH, inputC)
                found = True
                break

        if not found:
            print("WHAAAAAT?")
            raise ValueError("Beware.  Something has gone horribly wrong.  You may not have long to"
                             " live.")

        # Set our encoded vector as the root of the new tree
        rootNode = Node(newH)
        rootNode.children = [vec[0] for vec in children]
        return (rootNode, newC)

    def initialize_forget_bias(self, bias_value):
        for lstm in self.lstm_list:
            lstm.initialize_forget_bias(bias_value)

class BinaryTreeLSTM(nn.Module):
    '''
    BinaryTreeLSTM

    Takes in a binary tree where each node has a value and a list of children.
    Produces a tree of the same size where the value of each node is now encoded.
    '''

    def __init__(self, input_size, hidden_size):
        """
        Initialize tree cell we'll need later.
        """
        super(BinaryTreeLSTM, self).__init__()

        self.tree_lstm = TreeCell(input_size, hidden_size, 2)            
        self.register_buffer('zero_buffer', torch.zeros(hidden_size))

    def forward(self, node):
        """
        Creates a tree where each node's value is the encoded version of the original value.

        :param tree: a tree where each node has a value vector and a list of children
        :return a tuple - (root of encoded tree, cell state)
        """
        value = node.value
        
        if value is None:
            return (Node(None), self.zero_buffer)
        
        # List of tuples: (node, cell state)
        children = []

        # Recursively encode children
        for child in node.children:
            encoded_child = self.forward(child)
            children.append(encoded_child)

        # Extract the TreeCell inputs
        inputH = [vec[0].value for vec in children]
        inputC = [vec[1] for vec in children]

        for i, hidden in enumerate(inputH):
            if hidden is None:
                inputH[i] = self.zero_buffer
        
        if len(children) <= 2:
            newH, newC = self.tree_lstm(value, inputH + [self.zero_buffer]*(2 - len(children)), inputC + [self.zero_buffer]*(2 - len(children)))
        else:
            print("WHAAAAAT?")
            raise ValueError("Beware.  Something has gone horribly wrong.  You may not have long to"
                             " live.")

        # Set our encoded vector as the root of the new tree
        rootNode = Node(newH)
        rootNode.children = [vec[0] for vec in children]
        return (rootNode, newC)

    def initialize_forget_bias(self, bias_value):
        for lstm in self.lstm_list:
            lstm.initialize_forget_bias(bias_value)
