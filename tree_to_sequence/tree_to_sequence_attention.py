import torch
import torch.nn as nn

from tree_to_sequence.tree_to_sequence import TreeToSequence

class TreeToSequenceAttention(TreeToSequence):
    def __init__(self, encoder, decoder, hidden_size, nclass, embedding_size, 
                 alignment_size=50, align_type=1):
        super(TreeToSequenceAttention, self).__init__(encoder, decoder, hidden_size, nclass, 
                                                      embedding_size)

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

    """
        input: The output of the encoder for the tree should have be a triple. The first
               part of the triple should be the annotations and have dimensions,
               number_of_nodes x hidden_size. The second triple of the pair should be the hidden
               representations of the root and should have dimensions, num_layers x hidden_size.
               The third part should correspond to the cell states of the root and should
               have dimensions, num_layers x hidden_size.
        target: The target should have dimensions, seq_len, and should be a LongTensor.
    """
    def forward_train(self, input, target, teacher_forcing=True):
        annotations, decoder_hiddens, decoder_cell_states = self.encoder(input)
        # align_size: 0 number_of_nodes x alignment_size or 
        # align_size: 1-2 bengio number_of_nodes x hidden_size
        if self.align_type <= 1:
            attention_hidden_values = self.attention_hidden(annotations)
        else:
            attention_hidden_values = annotations

        decoder_hiddens = decoder_hiddens.unsqueeze(1) # num_layers x 1 x hidden_size
        decoder_cell_states = decoder_cell_states.unsqueeze(1) # num_layers x 1 x hidden_size

        target_length, = target.size()
        word_input = self.embedding(self.SOS_token).squeeze(0) # 1 x embedding_size
        loss = 0
        
        et = self.et
        
        for i in range(target_length):
            # 1 x embedding_size + hidden_size
            decoder_input = torch.cat((word_input, et), dim=1) 
            decoder_hiddens, decoder_cell_states = self.decoder(decoder_input, 
                                                                (decoder_hiddens, 
                                                                 decoder_cell_states))
            decoder_hidden = decoder_hiddens[-1]

            attention_logits = self.attention_logits(attention_hidden_values, decoder_hidden)
            attention_probs = self.softmax(attention_logits) # number_of_nodes x 1
            context_vec = (attention_probs * annotations).sum(0).unsqueeze(0) # 1 x hidden_size
            et = self.tanh(self.attention_presoftmax(torch.cat((decoder_hidden, context_vec), 
                                                               dim=1)))
            log_odds = self.output_log_odds(et)
            loss += self.loss_func(log_odds, target[i].unsqueeze(0))

            if teacher_forcing:
                next_input = target[i].unsqueeze(0)
            else:
                _, next_input = log_odds.topk(1)

            word_input = self.embedding(next_input).squeeze(1) # 1 x embedding size
        return loss

    """
        This is just an alias for point_wise_prediction, so that training code that assumes the 
        presence of a forward_train and forward_prediction works.
    """
    def forward_prediction(self, input, maximum_length=150):
        return self.point_wise_prediction(input, maximum_length)

    def point_wise_prediction(self, input, maximum_length=150):
        annotations, decoder_hiddens, decoder_cell_states = self.encoder(input)

        # align_size: 0 number_of_nodes x alignment_size or 
        # align_size: 1-2 bengio number_of_nodes x hidden_size
        if self.align_type <= 1:
            attention_hidden_values = self.attention_hidden(annotations)
        else:
            attention_hidden_values = annotations

        decoder_hiddens = decoder_hiddens.unsqueeze(1) # num_layers x 1 x hidden_size
        decoder_cell_states = decoder_cell_states.unsqueeze(1) # num_layers x 1 x hidden_size

        word_input = self.embedding(self.SOS_token).squeeze(0) # 1 x embedding_size
        output_so_far = []
        
        et = self.et

        for i in range(maximum_length):
            # 1 x embedding_size + hidden_size
            decoder_input = torch.cat((word_input, et), dim=1)
            decoder_hiddens, decoder_cell_states = self.decoder(decoder_input, 
                                                                (decoder_hiddens, 
                                                                 decoder_cell_states))
            decoder_hidden = decoder_hiddens[-1]

            attention_logits = self.attention_logits(attention_hidden_values, decoder_hidden)
            attention_probs = self.softmax(attention_logits) # number_of_nodes x 1
            context_vec = (attention_probs * annotations).sum(0).unsqueeze(0) # 1 x hidden_size
            et = self.tanh(self.attention_presoftmax(torch.cat((decoder_hidden, context_vec), 
                                                               dim=1)))
            log_odds = self.output_log_odds(et)
            _, next_input = log_odds.topk(1)

            output_so_far.append(int(next_input))

            if int(next_input) == self.EOS_value:
                break

            word_input = self.embedding(next_input).squeeze(1) # 1 x embedding size

        return output_so_far

    def beam_search_prediction(self, input, maximum_length=20, beam_width=5):
        annotations, decoder_hiddens, decoder_cell_states = self.encoder(input)
        # align_size: 0 number_of_nodes x alignment_size or 
        # align_size: 1-2 bengio number_of_nodes x hidden_size
        if self.align_type <= 1:
            attention_hidden_values = self.attention_hidden(annotations)
        else:
            attention_hidden_values = annotations

        decoder_hiddens = decoder_hiddens.unsqueeze(1) # num_layers x 1 x hidden_size
        decoder_cell_states = decoder_cell_states.unsqueeze(1) # num_layers x 1 x hidden_size

        word_input = self.embedding(self.SOS_token).squeeze(0) # 1 x embedding_size
        decoder_input = torch.cat((word_input, self.et), dim=1)
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
                decoder_hiddens, decoder_cell_states = self.decoder(decoder_input, 
                                                                    (decoder_hiddens, 
                                                                     decoder_cell_states))
                decoder_hidden = decoder_hiddens[-1]

                attention_logits = self.attention_logits(attention_hidden_values, decoder_hidden)
                attention_probs = self.softmax(attention_logits) # number_of_nodes x 1
                context_vec = (attention_probs * annotations).sum(0).unsqueeze(0) # 1 x hidden_size
                et = self.tanh(self.attention_presoftmax(torch.cat((decoder_hidden, context_vec), 
                                                                   dim=1))) # 1 x hidden_size
                log_odds = self.output_log_odds(et).squeeze(0) # nclasses
                log_probs = self.log_softmax(log_odds)

                log_value, next_input = log_probs.topk(beam_width) # beam_width, beam_width
                word_input = self.embedding(next_input.unsqueeze(1)) 
                decoder_input = torch.cat((word_input, et.unsqueeze(0).repeat(beam_width, 1, 1)), 
                                          dim=2)

                new_word_inputs.extend((word_inputs[i][0] + float(log_value[k]), 
                                        word_inputs[i][1] + [int(next_input[k])],
                                        int(next_input[k]) != self.EOS_value, 
                                        [word_input[k], decoder_hiddens, decoder_cell_states])
                                        for k in range(beam_width))
            word_inputs = sorted(new_word_inputs, key=lambda word_input:word_input[0])[-beam_width:]
        return word_inputs[-1][1]

    def attention_logits(self, attention_hidden_values, decoder_hidden):
        if self.align_type == 0:
            return self.attention_alignment_vector(self.tanh(self.attention_context(decoder_hidden) 
                                                             + attention_hidden_values))
        else:
            return (decoder_hidden * attention_hidden_values).sum(1).unsqueeze(1)
