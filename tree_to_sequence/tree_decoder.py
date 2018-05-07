import torch
import torch.nn as nn

from tree_to_sequence.translating_trees import Node

class TreeDecoder(nn.Module):
    """
    Produces a tree from the hidden state and cell state of an encoded version of a tree.
    """
    def __init__(self, embedding_size, hidden_size, max_num_children, possible_nodes, align_type=1, max_depth=10):
        super(TreeDecoder, self).__init__()
        
        self.softmax = nn.Softmax(0)
        self.lstm_list = nn.ModuleList()
        for i in range(max_num_children):
            self.lstm_list.append(nn.LSTMCell(embedding_size + hidden_size, hidden_size))
                    
        self.prediction_matrix = nn.Linear(hidden_size, len(possible_nodes) + 1, bias=False)
        self.embedding = nn.Embedding(len(possible_nodes), embedding_size)
        
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
        self.EOS = len(possible_nodes) + 1

    def forward(self, rootH, rootC, annotations):

        new_tree = Node((rootH, rootC))

        # Create stack of unexpanded nodes
        unexpanded = [(new_tree, 10)]

        # while stack isn't empty:
        while (len(unexpanded)) > 0:
            # Pop last item
            node, height = unexpanded.pop()
            nodeH = node.value[0]
            nodeC = node.value[1]

            # Compute e_t
            if self.align_type <= 1:
                attention_hidden_values = self.attention_hidden(annotations)
            else:
                attention_hidden_values = annotations
            attention_logits = self.attention_logits(attention_hidden_values, nodeH)
            attention_probs = self.softmax(attention_logits) # number_of_nodes x 1
            context_vec = (attention_probs * annotations).sum(0).unsqueeze(0) # 1 x hidden_size
            et = self.tanh(self.attention_presoftmax(torch.cat((nodeH, context_vec), dim=1))) # 1 x hidden_size

            # t = argmax softmax(W * et) 
            _, prediction = torch.max(self.prediction_matrix(et), 1)
            node.value = (nodeH, nodeC, prediction)
            if int(prediction) == self.EOS:
                print("END")
            if not int(prediction) == self.EOS and height != 0: 
                for i in range(len(self.lstm_list)):
                    word_embedding = torch.cat((self.embedding(prediction), et), 1)
                    childH, childC = self.lstm_list[i](word_embedding, (nodeH, nodeC))
                    childNode = Node((childH, childC))
                    node.children.append(childNode)
                    unexpanded.append((childNode, height - 1))
        return new_tree

    def attention_logits(self, attention_hidden_values, decoder_hidden):
        if self.align_type == 0:
            return self.attention_alignment_vector(self.tanh(self.attention_context(decoder_hidden) + attention_hidden_values))
        else:
            return (decoder_hidden * attention_hidden_values).sum(1).unsqueeze(1)            

    def initialize_forget_bias(self, bias_value):
        for lstm in self.lstm_list:
            nn.init.constant(lstm.bias_ih, bias_value)
            nn.init.constant(lstm.bias_hh, bias_value)
