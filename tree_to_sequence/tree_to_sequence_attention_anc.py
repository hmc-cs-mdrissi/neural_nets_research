import torch
import torch.nn as nn
from torch.autograd import Variable

from ANC.Controller import Controller
from tree_to_sequence.tree_to_sequence_attention import TreeToSequenceAttention

class TreeToSequenceAttentionANC(TreeToSequenceAttention):
    def __init__(self, encoder, decoder, hidden_size, embedding_size, M, R,
                 N=11, alignment_size=50, align_type=1, correctness_weight=1,
                 halting_weight=1, confidence_weight=2, efficiency_weight=0, t_max=7):
        # The 1 is for nclasses which is not used in this model.
        super(TreeToSequenceAttentionANC, self).__init__(encoder, decoder, hidden_size, 1, embedding_size,
                                                         alignment_size=alignment_size, align_type=align_type)
        # the initial registers all have value 0 with probability 1
        prob_dist = torch.zeros(R, M)
        prob_dist[:, 0] = 1

        self.register_buffer('initial_registers', prob_dist)

        self.M, self.R, self.N = M, R, N
        self.correctness_weight, self.halting_weight = correctness_weight, halting_weight
        self.confidence_weight, self.efficiency_weight = confidence_weight, efficiency_weight
        self.t_max = t_max

        self.initial_word_input = nn.Parameter(torch.Tensor(1, N + 3*R))
        self.output_log_odds = nn.Linear(hidden_size, N + 3*R)


    """
        input: The output of the encoder for the tree should have be a triple. The first
               part of the triple should be the annotations and have dimensions,
               number_of_nodes x hidden_size. The second triple of the pair should be the hidden
               representations of the root and should have dimensions, num_layers x hidden_size.
               The third part should correspond to the cell states of the root and should
               have dimensions, num_layers x hidden_size.
        target: The target should be a list of triples, where the first element of any triple is
                the input matrix, the second element is the output matrix corresponding to the expected
                output based on the input and the third element is a mask that specifies the area
                of memory where the output is.
    """
    def forward(self, input, target):
        controller = self.forward_prediction(input)
        return self.compute_loss(controller, target)

    """
        input: The output of the encoder for the tree should have be a triple. The first
               part of the triple should be the annotations and have dimensions,
               number_of_nodes x hidden_size. The second triple of the pair should be the hidden
               representations of the root and should have dimensions, num_layers x hidden_size.
               The third part should correspond to the cell states of the root and should
               have dimensions, num_layers x hidden_size.
    """
    def forward_prediction(self, input):
        annotations, decoder_hiddens, decoder_cell_states = self.encoder(input)
        # align_size: 0 number_of_nodes x alignment_size or align_size: 1-2 bengio number_of_nodes x hidden_size
        if self.align_type <= 1:
            attention_hidden_values = self.attention_hidden(annotations)
        else:
            attention_hidden_values = annotations

        decoder_hiddens = decoder_hiddens.unsqueeze(1) # num_layers x 1 x hidden_size
        decoder_cell_states = decoder_cell_states.unsqueeze(1) # num_layers x 1 x hidden_size

        word_input = self.initial_word_input # 1 x N + 3*R
        et = Variable(self.et)

        output_words = []

        for i in range(self.M):
            decoder_input = torch.cat((word_input, et), dim=1) # 1 x N + 3*R + hidden_size
            decoder_hiddens, decoder_cell_states = self.decoder(decoder_input, (decoder_hiddens, decoder_cell_states))
            decoder_hidden = decoder_hiddens[-1]

            attention_logits = self.attention_logits(attention_hidden_values, decoder_hidden)
            attention_probs = self.softmax(attention_logits) # number_of_nodes x 1
            context_vec = (attention_probs * annotations).sum(0).unsqueeze(0) # 1 x hidden_size
            et = self.tanh(self.attention_presoftmax(torch.cat((decoder_hidden, context_vec), dim=1)))
            word_input = self.output_log_odds(et)
            output_words.append(word_input)

        controller_params = torch.stack(output_words, dim=2).squeeze(0) # N + 3*R x M
        instruction = controller_params[0:self.N]
        first_arg = controller_params[self.N:self.N+self.R]
        second_arg = controller_params[self.N+self.R:self.N+2*self.R]
        output = controller_params[self.N+2*self.R:self.N + 3*self.R]
        controller = Controller(first_arg=first_arg, second_arg=second_arg, output=output,
                                instruction=instruction, initial_registers=Variable(self.initial_registers),
                                multiplier=1, correctness_weight=self.correctness_weight, halting_weight=self.halting_weight,
                                confidence_weight=self.confidence_weight, efficiency_weight=self.efficiency_weight, t_max=self.t_max)

        if controller_params.is_cuda:
            controller = controller.cuda()

        return controller

    """
        controller: The controller for an ANC.
        target: The target should be a list of triples, where the first element of any triple is
                the input matrix, the second element is the output matrix corresponding to the expected
                output based on the input and the third element is a mask that specifies the area
                of memory where the output is.
    """
    def compute_loss(self, controller, target):
        loss = 0
        input_memories = target[0]
        output_memories = target[1]
        output_masks = target[2]

        for i in range(len(input_memories)):
            loss += controller.forward_train(input_memories[i], (output_memories[i], output_masks[i]))

        return loss/len(input_memories)
