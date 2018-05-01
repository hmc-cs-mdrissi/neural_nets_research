import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class NTM_Memory(nn.Module):
    def __init__(self, address_count, address_dimension, batch_size):
        super(NTM_Memory, self).__init__()
        self.initial_memory = nn.Parameter(torch.zeros(1, address_count, address_dimension))
        self.batch_size = batch_size

        self.reset_parameters()
        self.initialize_state()

    def reset_parameters(self):
        _, N, M = self.initial_memory.size()
        stdev = 1 / np.sqrt(N + M)
        nn.init.uniform(self.initial_memory, -stdev, stdev)

    def initialize_state(self):
        self.memory = self.initial_memory.repeat(self.batch_size, 1, 1)

    def address_memory(self, key_vec, prev_address_vec, β, g, s, γ):
        EPSILON = 1e-16
        result = F.cosine_similarity((key_vec + EPSILON).unsqueeze(1).expand_as(self.memory),
                                     self.memory + EPSILON, dim=2)
        result = F.softmax(β * result, dim=1)
        result = g * result + (1 - g) * prev_address_vec
        result = torch.cat((result[:, 1:], result[:, :1]), 1) * s[:, 0:1] + result * s[:, 1:2] + \
                 torch.cat((result[:, -1:], result[:, :-1]), 1) * s[:, 2:3]

        #         result = result ** γ
        #         result = result / (result.sum(1, keepdim=True) + EPSILON)

        return result

    def read_memory(self, address_vec):
        return torch.bmm(self.memory.transpose(1, 2), address_vec.unsqueeze(2)).squeeze(2)

    def update_memory(self, address_vec, erase_vec, add_vec):
        self.memory = self.memory * (1 - torch.bmm(address_vec.unsqueeze(2), erase_vec.unsqueeze(1)))
        self.memory += torch.bmm(address_vec.unsqueeze(2), add_vec.unsqueeze(1))