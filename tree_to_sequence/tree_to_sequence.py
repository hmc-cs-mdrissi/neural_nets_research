import torch
import torch.nn as nn

class TreeToSequence(nn.Module):
    """
      For the decoder this expects something like an lstm cell or a gru cell and not an lstm/gru.
      Batch size is not supported at all. More precisely the encoder expects an input that does not
      appear in batches and most also output non-batched tensors.
    """
    def __init__(self, encoder, decoder, hidden_size, nclass, embedding_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        # nclass + 1 to include end of sequence.
        self.output_log_odds = nn.Linear(hidden_size, nclass+1)
        self.softmax = nn.Softmax(dim=0)
        self.log_softmax = nn.LogSoftmax(dim=0)

        self.register_buffer('SOS_token', torch.LongTensor([[nclass+1]]))
        self.EOS_value = nclass

        # nclass + 2 to include start of sequence and end of sequence.
        # n + 1 - start of sequence, end of sequence - n.
        # The first n correspond to the alphabet in order.
        self.embedding = nn.Embedding(nclass+2, embedding_size)
        self.loss_func = nn.CrossEntropyLoss()

    """
        input: The output of the encoder for the input should be a pair. The first part
               should correspond to the hidden state of the root. The second part
               should correspond to the cell state of the root. They both should be
               [num_layers, hidden_size].
        target: The target should have dimension, seq_len, and should be a LongTensor.
    """
    def forward_train(self, input, target, teacher_forcing=True):
        # root hidden state/cell state
        decoder_hiddens, decoder_cell_states = self.encoder(input) # num_layers x hidden_size
        decoder_hiddens = decoder_hiddens.unsqueeze(1)
        decoder_cell_states = decoder_cell_states.unsqueeze(1)

        target_length, = target.size()
        decoder_input = self.embedding(self.SOS_token).squeeze(0) # 1 x embedding_size
        loss = 0

        for i in range(target_length):
            decoder_hiddens, decoder_cell_states = self.decoder(decoder_input, 
                                                                (decoder_hiddens,
                                                                 decoder_cell_states))
            decoder_hidden = decoder_hiddens[-1] # 1 x hidden_size
            log_odds = self.output_log_odds(decoder_hidden)
            loss += self.loss_func(log_odds, target[i].unsqueeze(0))

            if teacher_forcing:
                next_input = target[i].unsqueeze(0)
            else:
                _, next_input = log_odds.topk(1)

            decoder_input = self.embedding(next_input).squeeze(1) # 1 x embedding_size

        return loss

    """
        This is just an alias for point_wise_prediction, so that training code that assumes the 
        presence of a forward_train and forward_prediction works.
    """
    def forward_prediction(self, input, maximum_length=20):
        return self.point_wise_prediction(input, maximum_length)

    def point_wise_prediction(self, input, maximum_length=20):
        decoder_hiddens, decoder_cell_states = self.encoder(input)
        decoder_hiddens = decoder_hiddens.unsqueeze(1)
        decoder_cell_states = decoder_cell_states.unsqueeze(1)

        decoder_input = self.embedding(self.SOS_token).squeeze(0) # 1 x embedding_size
        output_so_far = []

        for _ in range(maximum_length):
            decoder_hiddens, decoder_cell_states = self.decoder(decoder_input, 
                                                                (decoder_hiddens, 
                                                                 decoder_cell_states))
            decoder_hidden = decoder_hiddens[-1]
            log_odds = self.output_log_odds(decoder_hidden)

            _, next_input = log_odds.topk(1)
            output_so_far.append(int(next_input))

            if int(next_input) == self.EOS_value:
                break

            decoder_input = self.embedding(next_input).squeeze(1) # 1 x embedding size

        return output_so_far

    def beam_search_prediction(self, input, maximum_length=20, beam_width=5):
        decoder_hiddens, decoder_cell_states = self.encoder(input)
        decoder_hiddens = decoder_hiddens.unsqueeze(1)
        decoder_cell_states = decoder_cell_states.unsqueeze(1)

        decoder_input = self.embedding(self.SOS_token).squeeze(0) # 1 x embedding_size
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
                log_odds = self.output_log_odds(decoder_hidden).squeeze(0) # nclasses
                log_probs = self.log_softmax(log_odds)

                log_value, next_input = log_probs.topk(beam_width) # beam_width, beam_width
                decoder_input = self.embedding(next_input.unsqueeze(1)) 

                new_word_inputs.extend((word_inputs[i][0] + float(log_value[k]), 
                                        word_inputs[i][1] + [int(next_input[k])],
                                        int(next_input[k]) != self.EOS_value, 
                                        [decoder_input[k], decoder_hiddens, decoder_cell_states])
                                        for k in range(beam_width))

            word_inputs = sorted(new_word_inputs, key=lambda word_input:word_input[0])[-beam_width:]
        return word_inputs[-1][1]
