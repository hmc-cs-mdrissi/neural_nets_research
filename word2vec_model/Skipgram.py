
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')


# In[2]:


import nltk

def create_dict(word_list):
    word_dict = {}
    for word in word_list:
        if not word in word_dict:
            word_dict[word] = len(word_dict)
    return word_dict

raw_data = nltk.corpus.treebank.words()
word_list = [word.lower() for word in raw_data if word.isalpha()]
word_dict = create_dict(word_list)


# In[3]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot(data):
    plt.figure()
    plt.plot(data)


# In[17]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as d
import torch.nn.functional as F
import torch.optim as optim


class Skipgram(nn.Module):
    """
    Skipgram model

    Args:
        hidden_layer_size: The second dimension of the hidden layer
        vocab_size: The vocabulary size. This should be the size of your word dictionary.
    """
    def __init__(self, hidden_layer_size, vocab_size):
        super(Skipgram, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, hidden_layer_size)
        self.output_layer = nn.Linear(hidden_layer_size, vocab_size)
        
    def forward(self, input):
        input_times_embedding = self.embedding_layer(input)
        output = self.output_layer(input_times_embedding)
        return torch.squeeze(output)
    
def train(input, labels):
    optimizer.zero_grad()
    input = Variable(input).cuda()
    output = skipgram(input)
    labels.transpose(0, 1)
    loss = skipgram_loss(output, labels)
    loss.backward()
    optimizer.step()


def trainAll(sentences, context_size, word_dict):
    dataset = SkipgramDataset(sentences, context_size, word_dict)
    
    data_loader = d.DataLoader(dataset, batch_size = batch_size)
    i = 0
    for _ in range(50):
        for word, label in data_loader:
            if i % 100 == 0:
                print(i)
            train(word, Variable(label).cuda())
            i += 1

# In the form
# 0 - ('a', ['I', 'am', 'purple', 'moose'])
# 1 - ('purple', ['am', 'a', 'moose', 'that'])
class SkipgramDataset(d.Dataset):
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
    def __init__(self, text, context, lookup):
        # Convert all word to their indices
        indexes = [lookup[word] for word in text]
        
        # Create word_map
        self.word_list = [(torch.LongTensor([indexes[i]]), 
                           torch.LongTensor(indexes[i-context: i] + indexes[i+1: i+1+context])) 
                          for i in range(context, len(text) - context)] 
        
    def __len__(self):
        return len(self.word_list)
    
    def __getitem__(self, i):
        return self.word_list[i]
        

    
def skipgram_loss(output, label):
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
    all_losses.append(total_loss.data[0])
    return total_loss
    
    
hidden_size = 300
batch_size = 1024

skipgram = nn.DataParallel(Skipgram(hidden_size, len(word_dict)).cuda())
optimizer = optim.SGD(skipgram.parameters(), lr = 0.001, momentum=0.9)
all_losses = []
context_size = 2
trainAll(word_list, context_size, word_dict)

    
        
        


# In[20]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses[0:-1:50])


# In[163]:


print(all_losses[0:-1:100])


# In[134]:




