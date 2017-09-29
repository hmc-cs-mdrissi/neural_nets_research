"""
    Defines the skipgram model for word2vec. The skipgram model can use either regular softmax
    or hierarchical softmax. Also contains a conveninent way of making a dataset for this
    model along with a loss function for both forms of the model.
"""

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

from hierarchical_softmax import HierarchicalSoftmax

class Skipgram(nn.Module):
    """
    Skipgram model

    Args:
        hidden_layer_size: The second dimension of the hidden layer
        vocab_size: The vocabulary size. This should be the size of your word dictionary.
    """
    def __init__(self, hidden_layer_size, vocab_size, huffman_tree=None):
        super(Skipgram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_layer_size, sparse=True)

        if huffman_tree is None:
            self.softmax_layer = nn.Linear(hidden_layer_size, vocab_size)
        else:
            self.softmax_layer = HierarchicalSoftmax(huffman_tree)

    def forward(self, input, id_list=None):
        if id_list is None:
            word_vector = self.embeddings(input).squeeze(1)
            probabilities = self.softmax_layer(word_vector)
        else:
            word_vector = self.embeddings(input).squeeze()
            probabilities = self.softmax_layer(word_vector, id_list.squeeze())

        return probabilities

    def lookup(self, word, word_dictionary):
        """
        Extracts the word vector for a word given the word and a dictionary that converts
        words to word ids.

        Args:
            word: The word whose vector you want.
            word_dictionary: A dictionary from words to id numbers.
        """
        word_id = word_dictionary[word]
        start_vec = Variable(torch.LongTensor([word_id]).unsqueeze(0)).cuda()

        return self.embeddings(start_vec).squueze()

    def backprop(self, id_list, lr):
        """
            Applies stochastic gradient descent to the weights that involve the id_list. Backwards
            should have been called before this. The reason to use this instead of an optimizer is
            to avoid iterating over all parameters.
        """
        if not isinstance(self.softmax_layer, HierarchicalSoftmax):
            raise ValueError('You can only call backprop when using hierarchical softmax.')

        self.softmax_layer.backprop(id_list, lr)

        for p in self.embeddings.parameters():
            p.data = p.data + (-lr) * p.grad.data
            # zero gradients after we make the calculation
            p.grad.data.zero_()


# In the form
# 0 - ('a', ['I', 'am', 'purple', 'moose'])
# 1 - ('purple', ['am', 'a', 'moose', 'that'])
class SkipgramDataset(data.Dataset):
    """
    Creates a Dataset for a Skipgram model. Each element in the dataset is formatted as
    [word, label], where word and both label are LongTensors. "word" is a certain element
    in the array of words passed in, and "label" is a list context_size nearby words to the
    left and right of "word".

    Args:
        text: A list of words which make up sentences. Punctuation words and capitalization
        may be included (although they will be removed).

        context_size: a positive integer representing the number of elements in either
        direction which may be paired with a word at a certain index.

        word_dict: a dictionary of words in the form
        word_dict = {
            "word_1": 0,
            "another_word": 1,
            "normal_word": 2
            ...
        }

    """
    def __init__(self, text, context_size, word_dictionary):
        # Convert all word to their indices
        indexes = [word_dictionary[word] for word in text]

        # Create word_map
        self.word_list = [(torch.LongTensor([indexes[i]]),
                           torch.LongTensor(indexes[i-context_size: i] + \
                                            indexes[i+1: i+1+context_size]))
                          for i in range(context_size, len(text) - context_size)]

    def __len__(self):
        return len(self.word_list)

    def __getitem__(self, i):
        return self.word_list[i]

def skipgram_cross_entropy_loss(output, label):
    """
    Computes the cross-entropy loss for a certain output tensor and a label by summing the
    cross_entropy loss values calculated for each colum [[AAAAA]]
    The output and label should have identical first-dimension sizes.

    Args:
        output: The output from a single pass through the Skipgram model. It should have dimension
        [batch_size x vocab_size] (where vocab_size is the size of the raw data's vocabulary).

        label: A variable with 0th dimesnsion equal to the 0th dimension of output.
        Each column represents the indices of an array of words which were nearby a certain other
        word.

    """
    total_loss = 0
    label = torch.transpose(label, 0, 1)
    for word in label:
        new_loss = F.cross_entropy(output, word)
        total_loss += new_loss

    return total_loss
