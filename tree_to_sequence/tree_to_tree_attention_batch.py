import torch
import torch.nn as nn
from tree_to_sequence.translating_trees import ( Node, pretty_print_tree )
from tree_to_sequence.translating_trees import map_tree
# from tree_to_sequence.fold import Fold
from torchfold import Fold

class TreeToTreeAttentionBatch(nn.Module):
    def __init__(self, encoder, decoder, hidden_size, embedding_size, nclass,
                 root_value=-1, alignment_size=50, align_type=1, max_size=50):
        """
        Translates an encoded representation of one tree into another
        """
        super(TreeToTreeAttentionBatch, self).__init__()
        
        # Save useful values
        self.nclass = nclass
        self.encoder = encoder
        self.decoder = decoder
        self.align_type = align_type
        self.max_size = max_size
        self.root_value = root_value
        
        # EOS is always the last token
        self.EOS_value = nclass
        
        # Useful functions
        self.softmax = nn.Softmax(0)
        self.tanh = nn.Tanh()
        
        # Set up attention
        if align_type == 0:
            self.attention_hidden = nn.Linear(hidden_size, alignment_size)
            self.attention_context = nn.Linear(hidden_size, alignment_size, bias=False)
            self.attention_alignment_vector = nn.Linear(alignment_size, 1)
        elif align_type == 1:
            self.attention_hidden = nn.Linear(hidden_size, hidden_size)
            
        self.attention_presoftmax = nn.Linear(2 * hidden_size, hidden_size)
        self.embedding = nn.Embedding(nclass + 1, embedding_size)  
    
    def calc_attention(self, decoder_hiddens, annotations):
        """
        :param decoder_hiddens: lstm hidden state
        :param annotations: concatenated hidden states of all nodes
        :returns et: attention vector
        """
        #TODO: Move this part to an outer func
        if self.align_type <= 1:
            attention_hidden_values = self.attention_hidden(annotations)
        else:
            attention_hidden_values = annotations
        
        decoder_hidden_expanded = decoder_hiddens.unsqueeze(1)
        attention_logits = self.attention_logits(attention_hidden_values, decoder_hidden_expanded)
        attention_probs = self.softmax(attention_logits) # number_of_nodes x 1
        context_vec = (attention_probs * annotations).sum(1).unsqueeze(1) #1 x 1 x hidden_size
        et = self.tanh(self.attention_presoftmax(torch.cat((decoder_hidden_expanded, context_vec), 
                                                       dim=2))) # 1 x hidden_size
        return et
    
    def calc_loss(self, parent, child_index, et, true_value):
        """
        Call the decoder's loss function.
        
        :param parent: node's parent (dummy; used for compatibility with grammar decoder)
        :param child_index: index of generated child (dummy; used for compatibility with grammar 
                            decoder)
        :param et: vector incorporating info from the attention and hidden state of past node
        :param true_value: true value of the new node
        :returns: cross entropy loss
        
        """
        return self.decoder.calculate_loss(parent, child_index, et, true_value)
    
    def get_next_decoder_input(self, next_input, et):
        """
        Concatenate a node's value with it's parent's attention vector.
        
        :param next_input: child node's value
        :param et: attention vector:
        :returns: concatenation of inputs
        """ 
        return torch.cat((self.embedding(next_input), et), 2)
        
    def get_next_child_states_left(self, parent, input, hidden_state, cell_state): # should be decoder specific
            return self.decoder.get_next_child_states_left(parent, input, hidden_state, cell_state)
        
    def get_next_child_states_right(self, parent, input, hidden_state, cell_state): # should be decoder specific
            return self.decoder.get_next_child_states_right(parent, input, hidden_state, cell_state)
        
    
    def plus(self, first, second):
        """
        Add two things, return their sum
        
        :param first: First addend
        :param second: Second addend
        :returns first + second
        """
        return first + second
    
    def attention_logits(self, attention_hidden_values, decoder_hidden):
        """
        Calculates the logits over the nodes in the input tree.
        
        :param attention_hidden_values: representation of attention to encoder values
        :param decoder_hidden: decoder hidden state
        :returns: attention logits
        """
        if self.align_type == 0:
            return self.attention_alignment_vector(self.tanh(self.attention_context(decoder_hidden) 
                                                             + attention_hidden_values))
        else:
            return (decoder_hidden * attention_hidden_values).sum(1).unsqueeze(1) 
        

    def forward_train(self, input_tree_list, target_tree_list, teacher_forcing=True):
        """
        Encodes nodes of a tree in the rows of a matrix.

        :param tree: a tree where each node has a value vector and a list of children
        :return a matrix where each row represents the encoded output of a single node and also
                the hidden/cell states of the root node.

        """
#         input_tree_list = [map_tree(lambda node: node.unsqueeze(0), tree) for tree in input_tree_list]
        target_tree_list = [map_tree(lambda node: node.unsqueeze(0), tree) for tree in target_tree_list]
        
#         # Encoding
#         fold = Fold()
#         self.encoder.embedding = self.embedding
        
        
#         result = [self.encoder(fold, tree) for tree in input_tree_list]
#         hiddens = [x[0] for x in result]
#         cell_states = [x[1] for x in result]
#         decoder_hiddens, decoder_cell_states = fold.apply(self, [hiddens, cell_states])
#         annotations = fold.get_all_computed("encode_node_with_children", 0)
        
#         # UGLY UGLY UGLY
#         annotations = annotations.squeeze(2)
#         annotations = annotations.unsqueeze(1)
        
#         print('ANNOTATIONS, DECODER_HIDDENS', annotations.shape, decoder_hiddens.shape)
        annotations_list = []
        decoder_hiddens_list = []
        decoder_cell_states_list = []
        
        for input_tree in input_tree_list:
            annotations, decoder_hiddens, decoder_cell_states = self.encoder(input_tree)
            annotations_list.append(annotations)
            decoder_hiddens_list.append(decoder_hiddens)
            decoder_cell_states_list.append(decoder_cell_states)
            
        annotations = torch.stack(annotations_list, 0).unsqueeze(1)
        decoder_hiddens = torch.stack(decoder_hiddens_list, 0)
        decoder_cell_states = torch.stack(decoder_cell_states_list, 0)
        
        fold2 = Fold()
        losses = [self.decode(fold2, hidden, cell_state, tree, -1, 0, annotation) for hidden, cell_state, tree, annotation in zip(decoder_hiddens, decoder_cell_states, target_tree_list, annotations)]        
        computed_losses = fold2.apply(self, [losses])[0]
        return torch.sum(computed_losses) / len(input_tree_list)
    
    def nothing(self, vec):
        return vec
    
    def unsqueeze(self, vec):
        return vec.unsqueeze(1)
                                                                      
    def unsqueeze2(self, vec):
        return vec.unsqueeze(1).unsqueeze(1) 
    
    def unsqueeze3(self, vec):
        return vec.unsqueeze(1).unsqueeze(1).unsqueeze(1) 
    
    def encode_none_node(self):
        """"
        Returns dummy zero vectors for the children of a node which doesn't have a child in that position

        :return annotations, hidden_state, cell_state
        """
        return self.encoder.encode_none_node()
    
    
    # TODO: Later make this stackable
    def encode_node_with_children(self, value, leftH, leftC, rightH, rightC):
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
        return self.encoder.encode_node_with_children(value, leftH, leftC, rightH, rightC)
    
    
    
    def make_tensor(self, x):
        return torch.tensor([x]).unsqueeze(0)
    
    def decode(self, fold, decoder_hiddens, decoder_cell_states, targetNode, parent_val, child_index, annotations): 
        """
        Generate predictions for an output tree given an input tree, then calculate the loss.
        
        :param fold: Torchfold object (used for batching)
        :param decoder_hiddens: lstm hidden state
        :param decoder_cell_states: lstm cell state
        :targetNode: correct node
        :parent_val: parent's value
        :annotations: hidden states of each node, as a matrix
        :returns: crossentropy loss of generated prediction
        
        
        Assumes teacher forcing!
        
        """
        et = fold.add("calc_attention", decoder_hiddens, annotations)
        loss = fold.add("calc_loss", self.make_tensor(parent_val), self.make_tensor(child_index), et, self.make_tensor(targetNode.value))
        next_input = targetNode.value
        decoder_input = fold.add("get_next_decoder_input", self.make_tensor(next_input), et)
        for i, child in enumerate(targetNode.children):
            # Parent node of a node's children is that node
            parent = next_input
            new_child_index = i
            if i == 0:
                func_name = "get_next_child_states_left"
            elif i == 1:
                func_name = "get_next_child_states_right"
            else:
                raise ValueError("Invalid child index")
            child_hiddens, child_cell_states = fold.add(func_name, torch.tensor(parent), 
                                                                     decoder_input, 
                                                                     decoder_hiddens, 
                                                                     decoder_cell_states).split(2)

            new_loss = self.decode(fold, child_hiddens, child_cell_states, child, parent, new_child_index, annotations)
            loss = fold.add("plus", loss, new_loss)

        return loss
    
    
    def forward_prediction(self, input_tree_list, output_tree_list):
        # Encoding
        fold = Fold()
        self.encoder.embedding = self.embedding
        result = [self.encoder(fold, tree) for tree in input_tree_list]
        annotations = [x[0] for x in result]
        hiddens = [x[1] for x in result]
        cell_states = [x[2] for x in result]
        annotations, hiddens, cell_states = fold.apply(self, [annotations, hiddens, cell_states])

        # Decoding
        fold2 = Fold()
        a_bigger = [fold2.add("nothing", vec) for vec in annotations]
        h_bigger = [fold2.add("unsqueeze2", vec) for vec in hiddens]
        c_bigger = [fold2.add("unsqueeze2", vec) for vec in cell_states]
        
        annotations_real, hiddens_real, cell_states_real = fold2.apply(self, [a_bigger, h_bigger, c_bigger])
        
        prediction_count = [self.predict(fold3, annotation, hidden, cell_state, input_tree, target_tree) for 
                            annotation, hidden, cell_state, input_tree, target_tree in
                           zip(annotations, hiddens, cell_states, input_tree_list, target_tree_list)]
        
        return torch.sum(prediction_count)
        
        
    
    def forward_prediction(self, input_tree, max_size=None):
        """
        Generate an output tree given an input tree
        
        :param input_tree: encoded input tree
        :param max_size: max. number of nodes to generate
        :returns: predicted output tree
        """
        if max_size is None:
            max_size = self.max_size
        
        # Encode tree
        annotations, decoder_hiddens, decoder_cell_states = self.encoder(input_tree)
        
        # Counter of how many nodes we've generated so far
        num_nodes = 0
        
        PLACEHOLDER = 1
        
        # Output tree we're building up.  The value is a placeholder
        tree = Node(PLACEHOLDER)
        
        # Create stack of unexpanded nodes
        # Tuple: (hidden_state, cell_state, desired_output, parent_value, child_index)
        unexpanded = [(decoder_hiddens, decoder_cell_states, tree, self.root_value, 0)]
        
        if self.align_type <= 1:
            attention_hidden_values = self.attention_hidden(annotations)
        else:
            attention_hidden_values = annotations
        
        # while stack isn't empty:
        while (len(unexpanded)) > 0:
            # Pop last item
            decoder_hiddens, decoder_cell_states, curr_root, parent_val, child_index = \
            unexpanded.pop()  
            
            # Use attention and pass hidden state to make a prediction
            decoder_hidden = decoder_hiddens[-1].unsqueeze(0)
            attention_logits = self.attention_logits(attention_hidden_values, decoder_hiddens)
            attention_probs = self.softmax(attention_logits) # number_of_nodes x 1
            context_vec = (attention_probs * annotations).sum(0).unsqueeze(0) # 1 x hidden_size
            et = self.tanh(self.attention_presoftmax(torch.cat((decoder_hiddens, context_vec), 
                                                               dim=1))) # 1 x hidden_size
            next_input = self.decoder.make_prediction(parent_val, child_index, et)
            curr_root.value = next_input
            num_nodes += 1
            
            # Only generate up to max_size nodes
            if num_nodes > self.max_size:
                break
            
            # If we have an EOS, there are no children to generate
            if int(curr_root.value) == self.EOS_value:
                continue
                
            decoder_input = torch.cat((self.embedding(next_input), et), 1)
            parent = next_input
            
            for i in range(self.decoder.number_children(parent)):                    
                # Get hidden state and cell state which will be used to generate this node's 
                # children
                child_hiddens, child_cell_states = self.decoder.get_next(parent, i, 
                                                                         decoder_input, 
                                                                         decoder_hiddens, 
                                                                         decoder_cell_states)
                # Add children to the stack
                curr_child = Node(PLACEHOLDER)
                unexpanded.append((child_hiddens, child_cell_states, curr_child, parent, i))
                curr_root.children.append(curr_child)
        return tree
    
    def print_example(self, input_tree, target_tree):
        """
        Print out the desired and actual output trees for one example
        
        :param input_tree: tree used to generate prediction
        :param target_tree: correct output tree
        """
        print("DESIRED")
        pretty_print_tree(target_tree)
        print("WE GOT!")
        pretty_print_tree(self.forward_prediction(input_tree))

    
    def update_max_size(self, max_size):
        self.max_size = max_size
        