import torch
import torch.nn as nn
from torch.autograd import Variable

class HierarchicalSoftmaxTree(nn.Module):
    def __init__(self, huffman_tree):
        super(HierarchicalSoftmaxTree, self).__init__()
        self.huffman_tree = huffman_tree
        self.modules = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()
        self.root = huffman_tree.root

        for linear_module in huffman_tree.getModules():
            self.modules.append(linear_module)

    def forward_on_path(self, input_word_vec, desired_output):
        modules_on_path = self.huffman_tree.getModulesOnPath(desired_output)
        probability = Variable(torch.ones(1), requires_grad=True)

        for module in modules_on_path:
            branch_probability = self.sigmoid(module(input_word_vec))
            probability *= branch_probability

        return probability

    def forward(self, input_word_vec, id_list):
        probability_list = []

        for batch in id_list:
            probability_list.append([self.forward_on_path(input_word_vec, desired_output)
                                     for desired_output in batch])

        return torch.FloatTensor(probability_list)

    def backprop_on_path(self, desired_output, learning_rate):
        modules_on_path = self.huffman_tree.getModulesOnPath(desired_output)

        for module in modules_on_path:
            for p in module.parameters():
                p -= learning_rate * p.grad
                # zero gradients after we make the calculation
                p.zero_grad()


    def backprop(self, id_list, learning_rate):
        for batch in id_list:
            for desired_output in batch:
                self.backProp_on_path(desired_output, learning_rate)

    @staticmethod
    def negative_log_likelihood(probability_list):
        sum_var = Variable(torch.zeros(1), requires_grad=True)
        count = 0.0

        for batch in probability_list:
            for probability in batch:
                sum_var += -1 * torch.log(probability)
                count += 1

        return sum_var / count
