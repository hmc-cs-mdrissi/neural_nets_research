import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

#from neural_turing_machine.ntm import NTM
from tree_to_sequence.tree_to_sequence_attention import TreeToSequenceAttention
from tree_to_sequence.tree_to_sequence import TreeToSequence

class TreeToSequenceAttentionNTM(TreeToSequenceAttention):
    def __init__(self, encoder, decoder, ntm, hidden_size, embedding_size,
                 alignment_size=50, align_type=1):
        super(TreeToSequenceAttention, self).__init__(encoder, decoder, hidden_size, 1, embedding_size)

        self.attention_presoftmax = nn.Linear(2 * hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        # Initialize ntm with batch_size 1, output size 1, 1 read 1 write head
        self.ntm = ntm
        if align_type == 0:
            self.attention_hidden = nn.Linear(hidden_size, alignment_size)
            self.attention_context = nn.Linear(hidden_size, alignment_size, bias=False)
            self.attention_alignment_vector = nn.Linear(alignment_size, 1)
        elif align_type == 1:
            self.attention_hidden = nn.Linear(hidden_size, hidden_size)

        self.align_type = align_type
        self.register_buffer('et', torch.zeros(1, hidden_size))
        self.counter = 0

    """
        input: The output of the encoder for the tree should have be a triple. The first
               part of the triple should be the annotations and have dimensions,
               number_of_nodes x hidden_size. The second triple of the pair should be the hidden
               representations of the root and should have dimensions, num_layers x hidden_size.
               The third part should correspond to the cell states of the root and should
               have dimensions, num_layers x hidden_size.
        target: The target should have dimensions, seq_len, and should be a LongTensor.
    """
    def forward_train(self, prog_inputs, targets):
        tree = prog_inputs[0]
        loss = 0.0
        for input, output in zip(prog_inputs[1], targets):
            prediction = self.forward_prediction((tree,input))[0][0]
            loss += (prediction - output)**2

        return loss



    """
        This is just an alias for point_wise_prediction, so that training code that assumes the presence
        of a forward_train and forward_prediction works.
    """
    def forward_prediction(self, prog_input, print_time = False):
        return self.point_wise_prediction(prog_input, print_time)

    def point_wise_prediction(self, prog_input, print_time=False):
        tree_input = prog_input[0]
        input_val = prog_input[1]
        annotations, decoder_hiddens, decoder_cell_states = self.encoder(tree_input)
        # align_size: 0 number_of_nodes x alignment_size or align_size: 1-2 bengio number_of_nodes x hidden_size
        if self.align_type <= 1:
            attention_hidden_values = self.attention_hidden(annotations)
        else:
            attention_hidden_values = annotations

        decoder_hiddens = decoder_hiddens.unsqueeze(1)  # num_layers x 1 x hidden_size
        decoder_cell_states = decoder_cell_states.unsqueeze(1)  # num_layers x 1 x hidden_size
        SOS_token = Variable(self.SOS_token)

        current_val = SOS_token  # 1 x embedding_size
        et = Variable(self.et)
        loss = 0
        self.ntm.reset_reads(input_val)
#         if print_time:
#             print("ABOUT TO PRINT A BUNCHA STUFF ===========================")
        for i in range(len(input_val)):

            decoder_input = torch.cat((input_val[i].unsqueeze(0),
                                       current_val.type(torch.FloatTensor), et), dim=1)  # 1 x 2 + hidden_size
            decoder_hiddens, decoder_cell_states = self.decoder(decoder_input, (decoder_hiddens, decoder_cell_states))
            decoder_hidden = decoder_hiddens[-1]

            attention_logits = self.attention_logits(attention_hidden_values, decoder_hidden)
            attention_probs = self.softmax(attention_logits)  # number_of_nodes x 1
            context_vec = (attention_probs * annotations).sum(0).unsqueeze(0)  # 1 x hidden_size
            et = self.tanh(self.attention_presoftmax(torch.cat((decoder_hidden, context_vec), dim=1)))
            current_val = self.ntm.forward_step(et)
#             if print_time:
#                 print("CURRENT VAL", current_val)
            #  Feed in ntm output, input, attention in each, input is 0 for all but k - 1
        return current_val

    def beam_search_prediction(self, input, maximum_length=20, beam_width=5):
        annotations, decoder_hiddens, decoder_cell_states = self.encoder(input)
        # align_size: 0 number_of_nodes x alignment_size or align_size: 1-2 bengio number_of_nodes x hidden_size
        if self.align_type <= 1:
            attention_hidden_values = self.attention_hidden(annotations)
        else:
            attention_hidden_values = annotations

        decoder_hiddens = decoder_hiddens.unsqueeze(1) # num_layers x 1 x hidden_size
        decoder_cell_states = decoder_cell_states.unsqueeze(1) # num_layers x 1 x hidden_size
        SOS_token = Variable(self.SOS_token)

        word_input = self.embedding(SOS_token).squeeze(0) # 1 x embedding_size
        et = Variable(self.et)

        decoder_input = torch.cat((word_input, et), dim=1)
        word_inputs = []

        for _ in range(beam_width):
            word_inputs.append((0, [], True, [decoder_input, decoder_hiddens, decoder_cell_states]))

        for _ in range(maximum_length):
            new_word_inputs = []

            for i in range(beam_width):
                if not word_inputs[i][2]:
                    new_word_inputs.append(word_inputs[i])
                    continue

                decoder_input, decoder_hiddens, decoder_cell_states = word_inputs[i][3]
                decoder_hiddens, decoder_cell_states = self.decoder(decoder_input, (decoder_hiddens, decoder_cell_states))
                decoder_hidden = decoder_hiddens[-1]

                attention_logits = self.attention_logits(attention_hidden_values, decoder_hidden)
                attention_probs = self.softmax(attention_logits) # number_of_nodes x 1
                context_vec = (attention_probs * annotations).sum(0).unsqueeze(0) # 1 x hidden_size
                et = self.tanh(self.attention_presoftmax(torch.cat((decoder_hidden, context_vec), dim=1))) # 1 x hidden_size
                log_odds = self.output_log_odds(et).squeeze(0) # nclasses
                log_probs = self.log_softmax(log_odds)

                log_value, next_input = log_probs.topk(beam_width) # beam_width, beam_width
                word_input = self.embedding(next_input.unsqueeze(1)) # beam_width x 1 x embedding size
                decoder_input = torch.cat((word_input, et.unsqueeze(0).repeat(beam_width, 1, 1)), dim=2)

                new_word_inputs.extend((word_inputs[i][0] + float(log_value[k]), word_inputs[i][1] + [int(next_input[k])],
                                        int(next_input[k]) != self.EOS_value, [word_input[k], decoder_hiddens, decoder_cell_states])
                                        for k in range(beam_width))
            word_inputs = sorted(new_word_inputs, key=lambda word_input: word_input[0])[-beam_width:]
        return word_inputs[-1][1]

    def attention_logits(self, attention_hidden_values, decoder_hidden):
        if self.align_type == 0:
            return self.attention_alignment_vector(self.tanh(self.attention_context(decoder_hidden) + attention_hidden_values))
        else:
            return (decoder_hidden * attention_hidden_values).sum(1).unsqueeze(1)



