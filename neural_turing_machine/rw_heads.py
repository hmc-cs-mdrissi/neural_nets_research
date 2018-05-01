import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def _split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results


class NTM_Head(nn.Module):
    def __init__(self, address_count, address_dimension,
                 controller_output_size):
        super(NTM_Head, self).__init__()

        self.controller_output_size = controller_output_size
        self.N = address_count
        self.M = address_dimension

    def is_read_head(self):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def initialize_state(self):
        raise NotImplementedError


class NTM_Read_Head(NTM_Head):
    def __init__(self, address_count, address_dimension, controller_output_size, batch_size):
        super(NTM_Read_Head, self).__init__(address_count, address_dimension, controller_output_size)

        self.read_parameters_lengths = [self.M, 1, 1, 3, 1]
        self.fc_read_parameters = nn.Linear(controller_output_size, sum(self.read_parameters_lengths))

        self.batch_size = batch_size

        self.reset_parameters()
        self.initialize_state()

    def reset_parameters(self):
        nn.init.xavier_uniform(self.fc_read_parameters.weight, gain=1.4)
        nn.init.normal(self.fc_read_parameters.bias, std=0.01)

        self.initial_address_vec = nn.Parameter(torch.zeros(self.N))
        self.initial_read = nn.Parameter(torch.randn(1, self.M) * 0.01)

    def initialize_state(self):
        self.prev_address_vec = self.initial_address_vec.clone()
        self.prev_read = self.initial_read.repeat(self.batch_size, 1)

    def is_read_head(self):
        return True

    def forward(self, x, memory):
        read_parameters = self.fc_read_parameters(x)

        key_vec, β, g, s, γ = _split_cols(read_parameters, self.read_parameters_lengths)
        β = F.softplus(β)
        g = F.sigmoid(g)
        s = F.softmax(s, dim=1)
        γ = 1 + F.softplus(γ)

        self.prev_address_vec = memory.address_memory(key_vec, self.prev_address_vec, β, g, s, γ)
        new_read = memory.read_memory(self.prev_address_vec)
        self.prev_read = new_read
        return new_read


class NTM_Write_Head(NTM_Head):
    def __init__(self, address_count, address_dimension, controller_output_size):
        super(NTM_Write_Head, self).__init__(address_count, address_dimension, controller_output_size)

        self.write_parameters_lengths = [self.M, 1, 1, 3, 1, self.M, self.M]
        self.fc_write_parameters = nn.Linear(controller_output_size, sum(self.write_parameters_lengths))

        self.reset_parameters()
        self.initialize_state()

    def reset_parameters(self):
        nn.init.xavier_uniform(self.fc_write_parameters.weight, gain=1.4)
        nn.init.normal(self.fc_write_parameters.bias, std=0.01)

        self.initial_address_vec = nn.Parameter(torch.zeros(self.N))

    def initialize_state(self):
        self.prev_address_vec = self.initial_address_vec.clone()

    def is_read_head(self):
        return False

    def forward(self, x, memory):
        write_parameters = self.fc_write_parameters(x)

        key_vec, β, g, s, γ, erase_vec, add_vec = _split_cols(write_parameters, self.write_parameters_lengths)
        β = F.softplus(β)
        g = F.sigmoid(g)
        s = F.softmax(s, dim=1)
        γ = 1 + F.softplus(γ)
        erase_vec = F.sigmoid(erase_vec)

        self.prev_address_vec = memory.address_memory(key_vec, self.prev_address_vec, β, g, s, γ)
        memory.update_memory(self.prev_address_vec, erase_vec, add_vec)