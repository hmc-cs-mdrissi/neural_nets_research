import torch
import torch.nn as nn
from torch.autograd import Variable

class HierarchicalSoftmax(nn.Module):
    def __init__(self, huffman_tree):
        super(HierarchicalSoftmax, self).__init__()
        self.huffman_tree = huffman_tree
        self.module_list = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()
        self.root = huffman_tree.root

        for linear_module in huffman_tree.get_modules():
            self.module_list.append(linear_module)

    def forward_on_path(self, input_word_vec, desired_output):
        modules_on_path = self.huffman_tree.get_modules_on_path(desired_output.data[0])
        probability = 1.0

        for module in modules_on_path:
            branch_probability = self.sigmoid(module(input_word_vec))
            probability = probability * branch_probability

        return probability

    def forward(self, input_word_vec, id_list):
        probability_list = []

        for desired_output in id_list:
            probability_list.append(self.forward_on_path(input_word_vec, desired_output))

        return torch.cat(probability_list)

    def backprop_on_path(self, desired_output, lr):
        modules_on_path = self.huffman_tree.get_modules_on_path(desired_output.data[0])

        for module in modules_on_path:
            for p in module.parameters():
                p.data -= lr * p.grad.data
                # zero gradients after we make the calculation
                p.grad.data.zero_()


    def backprop(self, id_list, lr):
        for desired_output in id_list:
            self.backprop_on_path(desired_output, lr)


def nll_cost(probabilities):
    return -1 * torch.sum(torch.log(probabilities))

def lr_scheduler(epoch, init_lr=0.001, lr_decay_epoch=7):
    """
    Returns the current learning rate given the epoch. This decays the learning rate
    by a factor of 0.1 every lr_decay_epoch epochs.
    """
    return init_lr * (0.1**(epoch // lr_decay_epoch))
