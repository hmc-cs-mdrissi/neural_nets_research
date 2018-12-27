import torch
import torch.nn as nn
from tree_to_sequence.translating_trees import ( Node, pretty_print_tree )
from tree_to_sequence.translating_trees import map_tree
from torchfold import Fold
import numpy as np
from matplotlib import pyplot as plt

class TreeToTreeAttentionBatch(nn.Module):
    def __init__(self, encoder, decoder, hidden_size, embedding_size, nclass,
                 root_value=-1, alignment_size=50, align_type=1, max_size=50, use_cuda=False):
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
        self.use_cuda = use_cuda
        
        # EOS is always the last token
        self.EOS_value = nclass
        
        # Useful functions
        self.softmax = nn.Softmax(0)
        self.batch_softmax = nn.Softmax(1)
        self.tanh = nn.Tanh()
        
        # Set up attention
        if align_type == 0:
            self.attention_hidden = nn.Linear(hidden_size, alignment_size)
            self.attention_context = nn.Linear(hidden_size, alignment_size, bias=False)
            self.attention_alignment_vector = nn.Linear(alignment_size, 1)
        elif align_type == 1:
            self.attention_hidden = nn.Linear(hidden_size, hidden_size)
#             print("align type 1")
#         print("attention hidden takes in", hidden_size, "spits out", hidden_size)
            
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
#             print("annotations should be batch x 209 or something x 512 or something")
#             print("annotations really is", annotations.shape)
            attention_hidden_values = self.attention_hidden(annotations)
            
        else:
            attention_hidden_values = annotations
        
        decoder_hidden_expanded = decoder_hiddens.unsqueeze(1)
        attention_logits = self.attention_logits(attention_hidden_values, decoder_hidden_expanded)
        attention_probs = self.batch_softmax(attention_logits) # number_of_nodes x 1
        context_vec = (attention_probs * annotations).sum(1).unsqueeze(1) #1 x 1 x hidden_size
#         print("annotations", annotations.shape, annotations[0,0,:5])
#         print("att hidd vals", attention_hidden_values.shape, attention_hidden_values[0,0,:5])
#         print("logits", attention_logits.shape, attention_logits[0,:5])
#         print("probs", attention_probs.shape, attention_probs[0,:5])
#         print("context vec", context_vec.shape, context_vec[0,0,:5])
#         print("decoder_hidden_expanded", decoder_hidden_expanded.shape, decoder_hidden_expanded[0,0,:5])
        temp1 = self.attention_presoftmax(torch.cat((decoder_hidden_expanded, context_vec), dim=2))
#         print("presoftmaxed", temp1.shape, temp1[0,0,:5])
        
        et = self.tanh(self.attention_presoftmax(torch.cat((decoder_hidden_expanded, context_vec), 
                                                       dim=2))) # 1 x hidden_size
#         print("full et", et.shape, et[0,0,:5])
#         assert False
        return et
    
    def calc_loss(self, parent, child_index, et, true_value, print_time):
        """
        Call the decoder's loss function.
        
        :param parent: node's parent (dummy; used for compatibility with grammar decoder)
        :param child_index: index of generated child (dummy; used for compatibility with grammar 
                            decoder)
        :param et: vector incorporating info from the attention and hidden state of past node
        :param true_value: true value of the new node
        :returns: cross entropy loss
        
        """
        return self.decoder.calculate_loss(parent, child_index, et, true_value, print_time)
    
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
            # We want the last index
            index = len(attention_hidden_values.shape) - 1
            return (decoder_hidden * attention_hidden_values).sum(index).unsqueeze(index) 
        

#     def forward_train_real(self, input_tree_list, target_tree_list, teacher_forcing=True):
#         annotations, decoder_hiddens, decoder_cell_states = self.encoder(input_tree_list)
        
        
    
    def forward_train(self, input_tree_list, target_tree_list, teacher_forcing=True):
        """
        Encodes nodes of a tree in the rows of a matrix.

        :param tree: a tree where each node has a value vector and a list of children
        :return a matrix where each row represents the encoded output of a single node and also
                the hidden/cell states of the root node.

        """
        specific_example = [self.find_example(tree) for tree in target_tree_list]
#         input_tree_list = [map_tree(lambda node: node.unsqueeze(0), tree) for tree in input_tree_list] #T2T
        target_tree_list = [map_tree(lambda node: node.unsqueeze(0), tree) for tree in target_tree_list]
        
        # kangaroo
        annotations, decoder_hiddens, decoder_cell_states = self.encoder(input_tree_list)
        annotations = annotations.transpose(0, 1)
#         print("TRAIN input, hiddens, annotations", input_tree_list.shape, decoder_hiddens.shape, annotations.shape)
#         print(decoder_hiddens[:3,:3])
#         assert False
#         print("SPECIFIC EXAMPLE", specific_example)
#         for i, our_example in enumerate(specific_example):
#             if our_example:
#                 print("TRAINING DECODER HIDDEN", decoder_hiddens[i][:7])

#         annotations_list = [] #T2T
#         decoder_hiddens_list = [] #T2T
#         decoder_cell_states_list = [] #T2T
        
#         for input_tree in input_tree_list #T2T
#             annotations, decoder_hiddens, decoder_cell_states = self.encoder(input_tree) #T2T
#             annotations_list.append(annotations) #T2T
#             decoder_hiddens_list.append(decoder_hiddens) #T2T
#             decoder_cell_states_list.append(decoder_cell_states) #T2T
            
#         annotations = torch.stack(annotations_list, 0).unsqueeze(1) #T2T
#         decoder_hiddens = torch.stack(decoder_hiddens_list, 0) #T2T
#         decoder_cell_states = torch.stack(decoder_cell_states_list, 0) #T2T
        
    
        # Expand the dimensions of each of our encoder outputs
        annotations = annotations.unsqueeze(1)
        decoder_hiddens = decoder_hiddens.unsqueeze(1)
        decoder_cell_states = decoder_cell_states.unsqueeze(1)
        
        
        fold2 = Fold()
        losses = [self.decode(fold2, hidden, cell_state, tree, -1, 0, annotation, bool_val) for hidden, cell_state, tree, annotation, bool_val in zip(decoder_hiddens, decoder_cell_states, target_tree_list, annotations, specific_example)] 
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
        tensor_x = torch.tensor([x]).unsqueeze(0)
        if self.use_cuda:
            return tensor_x.cuda()
        else:
            return tensor_x
    
    def decode(self, fold, decoder_hiddens, decoder_cell_states, targetNode, parent_val, child_index, annotations, print_time=False): 
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
        loss = fold.add("calc_loss", self.make_tensor(parent_val), self.make_tensor(child_index), et, self.make_tensor(targetNode.value), self.make_tensor(print_time))
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
                raise ValueError("Invalid child index %i" % i)
            child_hiddens, child_cell_states = fold.add(func_name, torch.tensor(parent), 
                                                                     decoder_input, 
                                                                     decoder_hiddens, 
                                                                     decoder_cell_states).split(2)

            new_loss = self.decode(fold, child_hiddens, child_cell_states, child, parent, new_child_index, annotations, print_time)
            loss = fold.add("plus", loss, new_loss)

        return loss
    
    

    
    def forward_prediction(self, input_tree, max_size=None, output=None):
        """
        Generate an output tree given an input tree
        """
#         print("we're forward predicting")
        
#         special_example = False
#         if not output is None:
#             print("we've got output", output.value)
#             special_example = self.find_example(output)
#         if special_example:
#             print("we got a special example")
        
        if max_size is None:
            max_size = self.max_size
        
        # Encode tree
        # penguin
        annotations, decoder_hiddens, decoder_cell_states = self.encoder(input_tree[:1], training=False)
#         print("TEST input, hiddens, annotations", input_tree[:1].shape, decoder_hiddens.shape, annotations.shape)
#         print(decoder_hiddens[:3,:3])
#         assert False
#         if special_example:
#             print("PREDICTION DECODER HIDDEN", decoder_hiddens[:7])

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
#         print("annotations", annotations.shape, annotations[0, :5])
        first = False #special_example
        # while stack isn't empty:
        while (len(unexpanded)) > 0:
            # Pop last item
            decoder_hiddens, decoder_cell_states, curr_root, parent_val, child_index = \
            unexpanded.pop()  
            
            # Use attention and pass hidden state to make a prediction
#             print("decoder hidden before", decoder_hiddens.shape)
            decoder_hidden = decoder_hiddens[-1].unsqueeze(0)
#             print("decoder hidden after", decoder_hidden.shape)
            attention_logits = self.attention_logits(attention_hidden_values, decoder_hiddens)
            attention_probs = self.softmax(attention_logits) # number_of_nodes x 1
            #print("at_hid_val", attention_hidden_values.shape, "dec_hid", decoder_hidden.shape)
            #print("at_log", attention_logits.shape)
#             print(attention_logits)
            context_vec = (attention_probs * annotations).sum(0).unsqueeze(0) # 1 x hidden_size
#             print("att hidd vals", attention_hidden_values.shape, attention_hidden_values[0,:5])
#             print("logits", attention_logits.shape, attention_logits[:5])
#             print("probs", attention_probs.shape, attention_probs[:5])
#             print("context vec", context_vec.shape, context_vec[0,:5])
#             print("decoder_hidden_expanded", decoder_hiddens.shape, decoder_hiddens[0,:5])
#             temp1 = self.attention_presoftmax(torch.cat((decoder_hiddens, context_vec), dim=1))
#             print("presoftmaxed", temp1.shape, temp1[0,:5])
        
        
#             print("context vec", context_vec.shape, annotations.shape, attention_probs.shape, decoder_hiddens.shape)
            #print("shapes", attention_probs.shape, annotations.shape,(attention_probs * annotations).shape, context_vec.shape)
            et = self.tanh(self.attention_presoftmax(torch.cat((decoder_hiddens, context_vec), 
                                                               dim=1))) # 1 x hidden_size
#             print("full et", et.shape, et[0,:5])
#             assert False
            next_input = self.decoder.make_prediction(parent_val, child_index, et, print_time=first)
#             first=False
            curr_root.value = next_input
            num_nodes += 1
            
            # Only generate up to max_size nodes
            if num_nodes > self.max_size:
                break
            
            # If we have an EOS, there are no children to generate
            if int(curr_root.value) == self.EOS_value:
                continue
                
            #print("dims", self.embedding(next_input).shape, et.shape)
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
    
    def find_example(self, tree):
        PLUS = 6 + 26 * 2 + 10
        # Check root
        return tree.value == PLUS
        
    
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

    def display_normally(self, pic, title=None):
        if not title is None:
            plt.title(title)
        pic = pic * 255.0
        plt.imshow(np.repeat(np.int0(pic)[:,:,np.newaxis], 3, axis=2))
        plt.show()
        
    def print_img_tree_example(self, img, trees, tokens=None):
        tree = trees[0]
        print("IMAGE")
        self.display_normally(img[0][0])
        print("WE WANTED")
        if tokens is None:
            pretty_print_tree(tree)
            print("WE GOT")
            pretty_print_tree(self.forward_prediction(img, output=tree))
        else:
            pretty_print_tree(tree, tokens)
            print("WE GOT")
            pretty_print_tree(self.forward_prediction(img, output=tree), tokens)
            
        
        
        
    def update_max_size(self, max_size):
        self.max_size = max_size
        