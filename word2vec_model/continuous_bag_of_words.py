import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable

from hierarchical_softmax import HierarchicalSoftmax

class CBOW(nn.Module):
    def __init__(self, vocab_size, hidden_size, huffman_tree=None):
        super(CBOW, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, hidden_size, sparse=True)

        if huffman_tree is None:
            self.softmax_layer = nn.Linear(hidden_size, 1)
        else:
            self.softmax_layer = HierarchicalSoftmax(huffman_tree)

    def forward(self, context, id_list=None):
        if id_list is None:
            context_vector = torch.sum(self.embeddings(context), dim=1)
            probabilities = self.softmax_layer(context_vector)
        else:
            context_vector = torch.sum(self.embeddings(context), dim=1).squeeze()
            probabilities = self.softmax_layer(context_vector, id_list.squeeze())

        return probabilities

    def lookup(self, word, word_dictionary):
        word_id = word_dictionary[word]
        start_vec = Variable(torch.LongTensor([word_id]).unsqueeze(0)).cuda()

        return self.embeddings(start_vec).squueze()

    def backprop(self, id_list, lr):
        if not isinstance(self.softmax_layer, HierarchicalSoftmax):
            raise ValueError('You can only call backprop when using hierarchical softmax.')

        self.softmax_layer.backprop(id_list, lr)

        for p in self.embeddings.parameters():
            p.data = p.data + (-lr) * p.grad.data
            # zero gradients after we make the calculation
            p.grad.data.zero_()

class CBOWDataset(data.Dataset):
    def __init__(self, text, context_size, word_dictionary):
        self.word_dictionary = word_dictionary
        self.context_size = context_size
        self.word_indices = torch.LongTensor(list(map(lambda word: word_dictionary[word], text)))

    def __len__(self):
        return len(self.word_indices) - 2*self.context_size

    def __getitem__(self, index):
        prior_words = self.word_indices[index:index+self.context_size]
        later_words = self.word_indices[index+self.context_size+1:index+2*self.context_size+1]

        return torch.cat((prior_words, later_words)), self.word_indices[index+self.context_size]
