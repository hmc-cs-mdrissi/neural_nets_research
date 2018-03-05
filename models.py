import torch.nn as nn

class CNN_Sequence_Extractor(nn.Module):
    def __init__(self, nchannels, nclass, leakyRelu=False):
        super(CNN_Sequence_Extractor, self).__init__()

        # Size of the kernel (image filter) for each convolutional layer.
        ks = [3, 3, 3, 3, 3, 3, 2]
        # Amount of zero-padding for each convoutional layer.
        ps = [1, 1, 1, 1, 1, 1, 0]
        # The stride for each convolutional layer. The list elements are of the form (height stride, width stride).
        ss = [(2,2), (2,2), (1,1), (2,1), (1,1), (2,1), (1,1)]
        # Number of channels in each convolutional layer.
        nm = [64, 128, 256, 256, 512, 512, 512]

        # Initializing the container for the modules that make up the neural network the neurel netowrk.
        cnn = nn.Sequential()

        # Represents a convolutional layer. The input paramter i signals that this is the ith convolutional layer. The user also has the option to set batchNormalization to True which will perform a batch normalization on the image after it has undergone a convoltuional pass. There is no output but this function adds the convolutional layer module created here to the sequential container, cnn.
        def convRelu(i, batchNormalization=False):
            nIn = nchannels if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('leaky_relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        # Creating the 7 convolutional layers for the model.
        convRelu(0)
        convRelu(1)
        convRelu(2, True)
        convRelu(3)
        convRelu(4, True)
        convRelu(5)
        convRelu(6, True)

        self.cnn = cnn

    def forward(self, input):
        output = self.cnn(input)
        _, _, h, _ = output.size()
        assert h == 1, "the height of conv must be 1"
        output = output.squeeze(2) # [b, c, w]
        output = output.permute(2, 0, 1) #[w, b, c]
        return output

class CRNN(nn.Module):

    def __init__(self, nchannels, nclass, nhidden, num_lstm_layers = 2, leakyRelu=False):
        super(CRNN, self).__init__()

        # Instantiating the convolutional and recurrent neural net layers as attributes of the CRNN module
        self.cnn = CNN_Sequence_Extractor(nchannels, nclass, leakyRelu)
        self.rnn = nn.LSTM(512, nhidden, num_lstm_layers, bidirectional=True)
        self.embedding = nn.Linear(nhidden * 2, nclass)

    # A forward pass through the CRNN. Takes a batch of images as input and produces a tensor corresponding to vertical slices of the image x batch size x predicted probability of membership to each class.
    def forward(self, input):
        # conv features
        conv = self.cnn(input)

        # A forward pass through the LSTM layers. Takes in a batch of inputs and passes them through the LSTM layers.
        recurrent, _ = self.rnn(conv)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class Sequence_to_Sequence_Model(nn.Module):
    """
      For the decoder this expects something like an lstm cell or a gru cell and not an lstm/gru.
      This assumes the encoder spits out something of the form sequence length, batch size,
      channels.
    """
    def __init__(self, encoder, decoder, hidden_size, nclass, embedding_size,
                 decoder_cell_state_shape=None, use_lstm=False, use_cuda=True):
        super(Sequence_to_Sequence_Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        #nclass + 2 to include end of sequence and trash
        self.output_log_probs = nn.Linear(hidden_size, nclass+2)
        self.softmax = nn.Softmax()

        self.SOS_token = Variable(torch.LongTensor([[0]]))
        self.EOS_value = 1

        self.use_cuda = use_cuda

        if self.use_cuda:
            self.SOS_token = self.SOS_token.cuda()

        self.embedding = nn.Embedding(nclass, embedding_size)

        self.use_lstm = use_lstm
        #nclass + 1 is the trash category to avoid penalties after target's EOS token
        self.loss_func = nn.CrossEntropyLoss(ignore_index=nclass+1)

        if use_lstm:
            self.decoder_initial_cell_state = torch.zeros(decoder_initial_cell_state)

    def forward_train(self, input, target, use_teacher_forcing=False):
        # encoded features
        encoded_features = self.encoder(input) # [w, b, c]
        decoder_hidden = encoded_features[-1, :, :]

        batch_size, target_length = target.size()
        decoder_input = self.embedding(self.SOS_token).squeeze(0).repeat(batch_size, 1)
        loss = 0

        if self.use_lstm:
            decoder_cell_state = self.decoder_initial_cell_state

        for i in range(target_length):
            if self.use_lstm:
                decoder_output, (decoder_hidden, decoder_cell_state) = self.decoder(decoder_input, (decoder_hidden, decoder_cell_state))
            else:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            log_probs = self.output_log_probs(decoder_output)
            loss += self.loss_func(log_probs, target[i])

            if use_teacher_forcing:
            	next_input = target[i]
            else:
                _, topi = log_probs.topk(1)
                next_input = topi[:, 0]
     
            decoder_input = self.embedding(next_input.unsqueeze(1)).squeeze(1)

        return loss

    """
      Inputs must be of batch size 1
    """
    def point_wise_prediction(self, input, maximum_length=20):
      # encoded features
      encoded_features = self.encoder(input).squeeze(1) # [w, c]
      decoder_hidden = encoded_features[-1, :]
      decoder_input = self.embedding(self.SOS_token).squeeze(0)
      output_so_far = []

      if self.use_lstm:
          decoder_cell_state = self.decoder_initial_cell_state

      for i in range(maximum_length):
          if self.use_lstm:
              decoder_output, (decoder_hidden, decoder_cell_state) = self.decoder(decoder_input, (decoder_hidden, decoder_cell_state))
          else:
              decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

          log_probs = self.output_log_probs(decoder_output)

          _, topi = log_probs.data.topk(1)
          ni = topi[0, 0]

          if ni == self.EOS_value:
              break

          output_so_far.append(ni)
          decoder_input = self.embedding(Variable([ni]).unsqueeze(1)).squeeze(1)

          if self.use_cuda:
              decoder_input = decoder_input.cuda()

      return output_so_far


    def beam_search_prediction(self, input, maximum_length=20):
      pass

class Sequence_to_Sequence_Attention_Model(Sequence_to_Sequence_Model):
    def __init__(self, encoder, decoder, hidden_size, nclass, embedding_size,
                 alignment_size, decoder_cell_state_shape=None, use_lstm=False):
        super(Sequence_to_Sequence_Attention_Model, self).__init__(encoder, decoder, hidden_size, nclass,
                                                                   decoder_cell_state_shape=decoder_cell_state_shape,
                                                                   use_lstm=use_lstm)

        self.attention_hidden = nn.Linear(hidden_size, alignment_size)
        self.attention_context = nn.Linear(hidden_size, alignment_size, bias=False)
        self.tanh = nn.Tanh()
        self.attention_alignment_vector = nn.Linear(encoded_size, 1)
        self.hidden_size = hidden_size

    """
        input: The output of the encoder for the input should have dimensions, (seq_len x batch_size x input_size)
        target: The target should have dimensions, (seq_len x batch_size), and should be a LongTensor.
    """
    def forward_train(self, input, target, use_teacher_forcing=False):
        # Think about what the dimensions should be in your case. Some of this code assumes batches are present.
        encoded_features = self.encoder(input) # [w, b, c]
        encoded_features.transpose_(0,1) # [b, w, c]

        attention_hidden_values = self.attention_hidden(encoded_features)

        decoder_hidden = encoded_features[:, 0, hidden_size//2:] # This needs to be tweaked to corresponded to the root.
        target_length, batch_size = target.size()
        word_input = self.embedding(self.SOS_token).repeat(batch_size, 1)

        loss = 0

        for i in range(target_length):
          attention_logits = self.attention_alignment_vector(self.attention_context(decoder_hidden).unsqueeze(1)  + attention_hidden_values).squeeze(2)
          attention_probs = self.softmax(attention_logits, 1) # B x W
          context_vec = (attention_probs.unsqueeze(2) * encoded_features).sum(1) # B x C
          decoder_input = torch.cat((word_input, context_vec), dim=1)

          if self.use_lstm:
            decoder_output, (decoder_hidden, decoder_cell_state) = self.decoder(decoder_input, (decoder_hidden, decoder_cell_state))
          else:
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

          log_probs = self.output_log_probs(decoder_output)
          loss += self.loss_func(log_probs, target[i, :])

          if use_teacher_forcing:
            word_input = self.embedding(target[i, :].unsqueeze(1)).squeeze(1)
          else:
            _, topi = log_probs.data.topk(1)
            ni = topi[0, 0]

            if ni == self.EOS_value:
              break

            word_input = self.embedding(Variable([ni]).unsqueeze(1)).squeeze(1)

            if self.use_cuda:
              word_input = word_input.cuda()

        return loss


    """
      Inputs must be of batch size 1
    """
    def point_wise_prediction(self, input, maximum_length=20):
      # encoded features
      encoded_features = self.encoder(input).squeeze(1) # [w, c]
      attention_hidden_values = self.attention_hidden(encoded_features)

      decoder_hidden = encoded_features[0, hidden_size//2:].unsqueeze(0) # This needs to be tweaked to corresponded to the root.
      word_input = self.embedding(self.SOS_token).squeeze(0)
      output_so_far = []

      if self.use_lstm:
          decoder_cell_state = self.decoder_initial_cell_state

      for i in range(maximum_length):
          attention_logits = self.attention_alignment_vector(self.attention_context(decoder_hidden) + attention_hidden_values).squeeze(1)
          attention_probs = self.softmax(attention_logits, 0) # W
          context_vec = (attention_probs.unsqueeze(1) * encoded_features).sum(0) # C
          decoder_input = torch.cat((word_input, context_vec.unsqueeze(0)), dim=1)

          if self.use_lstm:
              decoder_output, (decoder_hidden, decoder_cell_state) = self.decoder(decoder_input, (decoder_hidden, decoder_cell_state))
          else:
              decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

          log_probs = self.output_log_probs(decoder_output)

          _, topi = log_probs.data.topk(1)
          ni = topi[0, 0]

          if ni == self.EOS_value:
              break

          output_so_far.append(ni)
          word_input = self.embedding(Variable([ni]).unsqueeze(1)).squeeze(1)

          if self.use_cuda:
              word_input = word_input.cuda()

      return output_so_far

    def beam_search_prediction(self, input, maximum_length=20):
      pass
