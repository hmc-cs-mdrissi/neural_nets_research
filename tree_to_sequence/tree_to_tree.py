import torch
import torch.nn as nn
from torch.autograd import Variable
from tree_to_sequence.translating_trees import Node
from tree_to_sequence.translating_trees import print_tree
from tree_to_sequence.translating_trees import pretty_print_tree
from tree_to_sequence.translating_trees import pretty_print_attention_t2t


class TreeToTree(nn.Module):
    def __init__(self, encoder, decoder, hidden_size, embedding_size, nclass=32, max_size=50,
                 max_num_children=2, alignment_size=50, align_type=1, binarized=True):
        """
        Translates an encoded representation of one tree into another
        """
        super(TreeToTree, self).__init__()
        
        # Save useful values
        self.nclass = nclass
        self.encoder = encoder
        self.decoder = decoder
        self.max_size = max_size
        self.binarized = binarized
        self.max_num_children = max_num_children
        self.align_type = align_type
        
        # EOS is always the last token
        self.EOS_value = nclass
        
        # Useful functions
        self.loss_func = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(0)
        self.tanh = nn.Tanh()
        
        # Set up attention
        if align_type == 0:
            self.attention_hidden = nn.Linear(hidden_size, alignment_size)
            self.attention_context = nn.Linear(hidden_size, alignment_size, bias=False)
            self.attention_alignment_vector = nn.Linear(alignment_size, 1)
        elif align_type == 1:
            self.attention_hidden = nn.Linear(hidden_size, hidden_size)
        self.register_buffer('et', torch.zeros(1, hidden_size))
        self.attention_presoftmax = nn.Linear(2 * hidden_size, hidden_size)
        
        self.embedding = nn.Embedding(nclass + 1, embedding_size)
        
        # Used to let us print sample output at regular intervals
        self.i = 0 
        

    def forward_train(self, input_tree, target_tree, teacher_forcing=True):
        """
        Generate predictions for an output tree given an input tree, then calculate the loss.
        """
        # Encode tree
        annotations, decoder_hiddens, decoder_cell_states = self.encoder(input_tree)
        
        if self.align_type <= 1:
            attention_hidden_values = self.attention_hidden(annotations)
        else:
            attention_hidden_values = annotations

        # number to accumulate loss
        loss = 0
        
        # Tuple: (hidden_state, cell_state, desired_output, parent_value, child_index)
        unexpanded = [(decoder_hiddens, decoder_cell_states, target_tree, None, 0)]
        
        # Any line relating to this list was put in just for debugging purposes
        all_attention_probs = [] 
        
        # while stack isn't empty:
        while (len(unexpanded)) > 0:
            # Pop last item
            decoder_hiddens, decoder_cell_states, targetNode, parent_val, child_index = unexpanded.pop()
            # Use attention and past hidden state to generate scores
            attention_logits = self.attention_logits(attention_hidden_values, decoder_hiddens)
            attention_probs = self.softmax(attention_logits) # number_of_nodes x 1
            all_attention_probs.append(attention_probs) #TODO - take out!
            context_vec = (attention_probs * annotations).sum(0).unsqueeze(0) # 1 x hidden_size
            et = self.tanh(self.attention_presoftmax(torch.cat((decoder_hiddens, context_vec), dim=1))) # 1 x hidden_size
            # Calculate loss
            loss = loss + self.decoder.calculate_loss(parent_val, child_index, et, targetNode.value)
            # If we have an EOS, there are no children to generate
            if int(targetNode.value) == self.EOS_value:
                continue
            
            # Teacher forcing means we use the correct value (not the predicted value) of a node to generate its children
            if teacher_forcing:
                next_input = targetNode.value
            else:
                _, next_input = log_odds.topk(1)
            
            decoder_input = torch.cat((self.embedding(next_input), et), 1)
            
            for i, child in enumerate(targetNode.children):
                # Parent node of a node's children is that node
                parent = next_input
                new_child_index = i
                # ... unless you're the right child
                if self.binarized and i == 1:
                    parent = parent_val
                    new_child_index = child_index + 1
                    
                # Get hidden state and cell state which will be used to generate this node's children
                child_hiddens, child_cell_states = self.decoder.get_next(parent, new_child_index, decoder_input, decoder_hiddens, decoder_cell_states)
                unexpanded.append((child_hiddens, child_cell_states, child, parent, new_child_index))

        # Uncomment if you want to see where the attention is focusing as each node is generated
#         if self.i % 200 == 0:
#             pretty_print_attention_t2t(all_attention_probs, input_tree, target_tree)
        self.i += 1
        return loss
    
    
    def forward_prediction(self, input_tree):
        """
        Generate an output tree given an input tree
        """
        # Encode tree
        annotations, decoder_hiddens, decoder_cell_states = self.encoder(input_tree)
        
        # Counter of how many nodes we've generated so far
        num_nodes = 0
        
        PLACEHOLDER = 1
        
        # Output tree we're building up.  The value is a placeholder
        tree = Node(PLACEHOLDER)
        
        # Create stack of unexpanded nodes
        # Tuple: (hidden_state, cell_state, desired_output, parent_value, child_index)
        unexpanded = [(decoder_hiddens, decoder_cell_states, tree, None, 0)]
        
        if self.align_type <= 1:
            attention_hidden_values = self.attention_hidden(annotations)
        else:
            attention_hidden_values = annotations
        
        # while stack isn't empty:
        while (len(unexpanded)) > 0:
            # Pop last item
            decoder_hiddens, decoder_cell_states, curr_root, parent_val, child_index = unexpanded.pop()  
            
            # Use attention and pas hidden state to make a prediction
            attention_logits = self.attention_logits(attention_hidden_values, decoder_hiddens)
            attention_probs = self.softmax(attention_logits) # number_of_nodes x 1
            context_vec = (attention_probs * annotations).sum(0).unsqueeze(0) # 1 x hidden_size
            et = self.tanh(self.attention_presoftmax(torch.cat((decoder_hiddens, context_vec), dim=1))) # 1 x hidden_size
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
            
            for i in range(self.max_num_children):
                # Parent node of a node's children is that node
                parent = next_input
                new_child_index = i
                # ... unless you're the right child
                if self.binarized and i == 1:
                    parent = parent_val
                    new_child_index = child_index + 1
                    
                # Get hidden state and cell state which will be used to generate this node's children
                child_hiddens, child_cell_states = self.decoder.get_next(parent, new_child_index, decoder_input, decoder_hiddens, decoder_cell_states)
                # Add children to the stack
                curr_child = Node(PLACEHOLDER)
                unexpanded.append((child_hiddens, child_cell_states, curr_child, parent, new_child_index))
                curr_root.children.append(curr_child)
        return tree
    
    def print_example(self, input_tree, target_tree):
        """
        Print out the desired and actual output trees for one example
        """
        print("DESIRED")
        pretty_print_tree(target_tree)
        print("WE GOT!")
        pretty_print_tree(self.forward_prediction(input_tree))

            
    def attention_logits(self, attention_hidden_values, decoder_hidden):
        """
        Calculates something attention-related.  Which you probably figured from the function name.  Which I also figured from the function name.  If you know more, please complete these comments.
        """
        if self.align_type == 0:
            return self.attention_alignment_vector(self.tanh(self.attention_context(decoder_hidden) + attention_hidden_values))
        else:
            return (decoder_hidden * attention_hidden_values).sum(1).unsqueeze(1)            

         
        

