import torch.nn as nn

class SequenceEncoder(nn.Module):
    # If you are using an end of sequence token that should be accounted for in input_size.
    def __init__(self, input_size, hidden_size, num_layers, attention=True,
                 use_embedding=True, embedding_size=256):
        super(SeqEncoder, self).__init__()

        self.use_embedding = use_embedding

        if use_embedding:
            self.embedding = nn.Embedding(input_size, embedding_size)
            self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

        self.attention = attention

    def initialize_forget_bias(self, bias_val):
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(bias_val)

    def forward(self, input):
        if self.use_embedding:
            input = self.embedding(input)
        outputs, (hiddens, cell_states) = self.lstm(input.unsqueeze(1))
        outputs, hiddens, cell_states = outputs.squeeze(1), hiddens.squeeze(1), cell_states.squeeze(1)

        if self.attention:
            return outputs, hiddens, cell_states
        else:
            return hiddens, cell_states
