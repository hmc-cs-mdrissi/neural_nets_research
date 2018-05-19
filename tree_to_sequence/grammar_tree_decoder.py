import torch
import torch.nn as nn
from torch.autograd import Variable

class GrammarTreeDecoder(nn.Module):
    """
    Decoder which produces a tree.  It only generates child nodes which are syntactically valid.
    """
    def __init__(self, embedding_size, hidden_size, max_num_children, parent_to_category, num_categories, category_to_child, num_vars, num_ints, binarized=True):
        """
        :param embedding_size: length of the encoded representation of a node
        :param hidden_size: hidden state size
        :param max_num_children: max. number of children a node can have
        :param parent_to_category: function which takes in a parent node and a child index 
            and returns the category of inputs it can produce at that index
        :param num_categories: number of different output categories
        :param category_to_child: function which takes in a category and returns 
            the indices of the tokens in that category.
        :param num_vars: number of variables the program can use
        :param num_ints: number of ints the program can use
        :param binarized: whether the tree is binarized with the left child, right sibling representation.
        """
        super(GrammarTreeDecoder, self).__init__()
        
        # Store values we'll use later
        self.parent_to_category = parent_to_category
        self.category_to_child = category_to_child
        self.num_vars = num_vars
        self.num_ints = num_ints
        self.binarized = binarized
        
        self.loss_func = nn.CrossEntropyLoss()
        self.LEFT = 0
        self.RIGHT = 1
        
        # The next two ModuleLists are lists of lists.  Each element in the outer list 
        # corresponds to a specific child index (e.g. left child = 0, righ child = 1).  
        # Each element in the inner list corresponds to a particular category of possible 
        # next tokens (e.g. one of the EXPR tokens).
                
        # A list of lists of linear layers which will be used to generate predictions for a node's value.
        self.linear_lists = nn.ModuleList()
        # A list of lists of linear layers which will be used to generate the hidden states we will later
        # use to generate a node's children.
        self.lstm_lists = nn.ModuleList()
        
        # Loop through possible child indices
        for child in range(max_num_children):
            linear_list = nn.ModuleList()
            lstm_list = nn.ModuleList()
            # Loop through possible categories
            for category in range(num_categories):
                possible_children = category_to_child(category, self.num_vars, self.num_ints)
                if len(possible_children) is 0:
                    linear_list.append(None)
                else:
                    linear_list.append(nn.Linear(hidden_size, len(possible_children)))
                lstm_list.append(nn.LSTMCell(embedding_size + hidden_size, hidden_size))
                
            self.linear_lists.append(linear_list)
            self.lstm_lists.append(lstm_list)
       
    
    def calculate_loss(self, parent, child_index, vec, true_value):
        """
        Calculate the crossentropy loss from the probabilities the decoder assigns 
        to each syntactically valid child of a parent node.
        
        :param parent: an integer holding the value of the parent node whose child we're generating
        :param child_index: index of the child to be generated (int)
        :param vec: et vector incorporating info from the attention and hidden state of past node
        :param true_value: true value of the new node
        :returns: crossentropy loss
        """
        # Get predictions
        log_odds, possible_indices = self.get_log_odds(parent, child_index, vec)
        # Get the index of the true value within the list of possibilities
        true_index = Variable(torch.LongTensor([possible_indices.index(int(true_value))]))
        return self.loss_func(log_odds, true_index)
        
    
    def get_log_odds(self, parent, child_index, vec):
        """
        Calculate a score for each syntactically valid value which could be generated 
        by the given parent at the given index.
        
        :param parent: parent whose child we're generating (int)
        :param child_index: index of the child to be generated (int)
        :param vec: et vector incorporating info from the attention and hidden state of past node
        """
        # Get the category of outputs this parent generates at this child_index
        parent_category_index = self.parent_to_category(parent, child_index, self.num_vars, self.num_ints)
        # If our tree is binarized, the *real* child index might be large, 
        # but we only have separate linear layers for the left child and right child.
        if self.binarized:
            child_index = self.LEFT if child_index == self.LEFT else self.RIGHT
        # Calculate log odds
        log_odds = self.linear_lists[child_index][int(parent_category_index)](vec)
        # Generate a list of possible child values
        possible_indices = self.category_to_child(parent_category_index, self.num_vars, self.num_ints)
        return log_odds, possible_indices
   

    def make_prediction(self, parent, child_index, vec):
        """
        Predict a token for the next node
        
        :param parent: parent of the node to be generated
        :param child_index: index of the child to be generated
        :param vec: et vector incorporating info from the attention and hidden state of past node
        """
        log_odds, possible_indices = self.get_log_odds(parent, child_index, vec)
        _, max_index = torch.max(log_odds, 1)
        return Variable(torch.LongTensor([possible_indices[int(max_index)]]))
    
    
    def get_next(self, parent, child_index, input, hidden_state, cell_state):
        """
        Generate the hidden and cell states which will be used to generate the current node's children
        
        :param parent: parent whose child we just generated (int)
        :param child_index: index of the child we just generated (int)
        :param input: embedded reprentation of the node's parent
        :param hidden_state: hidden state generated by the parent's lstm
        :param cell_state: cell state generated by the parent's lstm
        """
        parent_category_index = int(self.parent_to_category(parent, child_index, self.num_vars, self.num_ints))
        # If our tree is binarized, the *real* child index might be large, 
        # but we only have separate linear layers for the left child and right child.
        if self.binarized:
            child_index = self.LEFT if child_index == self.LEFT else self.RIGHT
        return self.lstm_lists[child_index][parent_category_index](input, (hidden_state, cell_state))
    
    
    def initialize_forget_bias(self, bias_value):
        """
        Initialize the forget bias to a certain value.  TODO: I don't actually know why this is used.  If you, future reader of this code, have the answer, please insert comment here.
        
        :param bias_value: value the forget bias wil be set to
        """
        for lstm_list in self.lstm_lists:
            for lstm in lstm_list:
                nn.init.constant(lstm.bias_ih, bias_value)
                nn.init.constant(lstm.bias_hh, bias_value)
