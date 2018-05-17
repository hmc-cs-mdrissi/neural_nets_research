import torch
import torch.nn as nn
from torch.autograd import Variable

from tree_to_sequence.translating_trees import Node
from tree_to_sequence.translating_trees import pretty_print_attention

class TreeDecoder(nn.Module):
    """
    Produces a tree from the hidden state and cell state of an encoded version of a tree.
    """
    def __init__(self, embedding_size, hidden_size, max_num_children, nclass, align_type=1):
        super(TreeDecoder, self).__init__()
                
        self.softmax = nn.Softmax(0)
        self.softmax1 = nn.Softmax(1)
        self.lstm_list = nn.ModuleList()
        
        for i in range(max_num_children):
            self.lstm_list.append(nn.LSTMCell(embedding_size + hidden_size, hidden_size))
                    
        self.output_log_odds = nn.Linear(hidden_size, nclass + 1)
        self.embedding = nn.Embedding(nclass + 1, embedding_size)
        
        self.attention_presoftmax = nn.Linear(2 * hidden_size, hidden_size)
        self.tanh = nn.Tanh()

        if align_type == 0:
            self.attention_hidden = nn.Linear(hidden_size, alignment_size)
            self.attention_context = nn.Linear(hidden_size, alignment_size, bias=False)
            self.attention_alignment_vector = nn.Linear(alignment_size, 1)
        elif align_type == 1:
            self.attention_hidden = nn.Linear(hidden_size, hidden_size)

        self.align_type = align_type
        self.register_buffer('et', torch.zeros(1, hidden_size))
        self.EOS_value = nclass
        self.i = 0 # Used to let us print sample output at regular intervals
        
        self.loss_func = nn.CrossEntropyLoss()
        

    def forward_train(self, rootH, rootC, target, annotations, teacher_forcing=True):
        # Create stack of unexpanded nodes #TODO: get rid of depth in this tuple if we end up not using it.
        unexpanded = [(rootH, rootC, target, 1)]
        
        if self.align_type <= 1:
            attention_hidden_values = self.attention_hidden(annotations)
        else:
            attention_hidden_values = annotations
            
        loss = 0

        all_attention_probs = []
        
        # while stack isn't empty:
        while (len(unexpanded)) > 0:
            # Pop last item
            nodeH, nodeC, targetNode, depth = unexpanded.pop()

            attention_logits = self.attention_logits(attention_hidden_values, nodeH)
            attention_probs = self.softmax(attention_logits) # number_of_nodes x 1
            all_attention_probs.append(attention_probs) #TODO - take out!
            context_vec = (attention_probs * annotations).sum(0).unsqueeze(0) # 1 x hidden_size
            et = self.tanh(self.attention_presoftmax(torch.cat((nodeH, context_vec), dim=1))) # 1 x hidden_size
            log_odds = self.softmax1(self.output_log_odds(et))
            
            node_loss = self.loss_func(log_odds, targetNode.value)
            if (int(targetNode.value) == self.EOS_value):
                node_loss = node_loss / 5.0
            loss += node_loss# * depth
                
            if int(targetNode.value) == self.EOS_value:
                continue
            
            if teacher_forcing:
                next_input = targetNode.value
            else:
                _, next_input = log_odds.topk(1)
            
            decoder_input = torch.cat((self.embedding(next_input), et), 1)
            
            for i, child in enumerate(targetNode.children[::-1]):
                childH, childC = self.lstm_list[i](decoder_input, (nodeH, nodeC))
                unexpanded.append((childH, childC, child, depth + 1))
# Uncomment if you want to see where the attention is focusing as each node is generated
#         if self.i % 200 == 0:
#             pretty_print_attention(all_attention_probs, target)
        self.i += 1          
        return loss
    
    def forward_prediction(self, rootH, rootC, annotations, max_nodes=20):
        tree = Node(1)
        
        # Create stack of unexpanded nodes
        unexpanded = [(rootH, rootC, tree)]
        
        # Compute e_t
        if self.align_type <= 1:
            attention_hidden_values = self.attention_hidden(annotations)
        else:
            attenteion_hidden_values = annotations
        
        num_nodes = 0
        
        # while stack isn't empty:
        while (len(unexpanded)) > 0:
            # Pop last item
            nodeH, nodeC, curr_root = unexpanded.pop()  

            attention_logits = self.attention_logits(attention_hidden_values, nodeH)
            attention_probs = self.softmax(attention_logits) # number_of_nodes x 1
            context_vec = (attention_probs * annotations).sum(0).unsqueeze(0) # 1 x hidden_size
            #et = self.tanh(self.attention_presoftmax(torch.cat((nodeH, context_vec), dim=1))) # 1 x hidden_size
            log_odds = self.softmax1(self.output_log_odds(et))
            _, next_input = log_odds.topk(1)
            curr_root.value = int(next_input)
            
            num_nodes += 1
            
            if num_nodes > max_nodes:
                break
            
            if curr_root.value == self.EOS_value:
                continue
            next_input = next_input.squeeze(0)
            decoder_input = torch.cat((self.embedding(next_input), et), 1)
            
            for lstm in self.lstm_list:            
                childH, childC = lstm(decoder_input, (nodeH, nodeC))
                curr_child = Node(1)
                unexpanded.append((childH, childC, curr_child))
                curr_root.children.append(curr_child)
        
        return tree

    def attention_logits(self, attention_hidden_values, decoder_hidden):
        if self.align_type == 0:
            return self.attention_alignment_vector(self.tanh(self.attention_context(decoder_hidden) + attention_hidden_values))
        else:
            return (decoder_hidden * attention_hidden_values).sum(1).unsqueeze(1)            

    def initialize_forget_bias(self, bias_value):
        for lstm in self.lstm_list:
            nn.init.constant(lstm.bias_ih, bias_value)
            nn.init.constant(lstm.bias_hh, bias_value)
