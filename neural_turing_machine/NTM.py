import torch
import torch.nn as nn

from rw_heads import NTM_Read_Head, NTM_Write_Head
from ntm_memory import NTM_Memory



class NTM(nn.Module):
    def __init__(self, batch_size, controller_output_size, output_size,
                address_count, address_dimension, heads, controller=None, stand_alone=False):
        super(NTM, self).__init__()

        self.batch_size = batch_size

        # Initialize controller
        if stand_alone:
            self.controller = controller

        # Create output gate. No activation function is used with it because
        # I used BCEWithLogitsLoss which deals with the sigmoid in a more
        # numerically stable manner.
        self.outputGate = nn.Linear(controller_output_size, output_size)

        # Initialize memory
        self.memory = NTM_Memory(address_count, address_dimension, batch_size)

        # Construct the heads.
        self.heads = nn.ModuleList()

        for head_id in heads:
            if head_id == 0:
                self.heads.append(NTM_Read_Head(address_count, address_dimension,
                                                controller_output_size, batch_size))
            else:
                self.heads.append(NTM_Write_Head(address_count, address_dimension,
                                                 controller_output_size))
        self.initialize_state()

    def initialize_state(self):
        self.prev_reads = []

        for head in self.heads:
            head.initialize_state()

            if head.is_read_head():
                self.prev_reads.append(head.prev_read)

        self.memory.initialize_state()

    def reset_parameters(self):
        nn.init.xavier_uniform(self.outputGate.weight)
        nn.init.normal(self.outputGate.bias, std=0.01)

    def forward_step(self, controller_output):
        self.initialize_state()
        outputs = []

        self.prev_reads = []

        for head in self.heads:
            if head.is_read_head():
                self.prev_reads.append(head(controller_output, self.memory))
            else:
                head(controller_output, self.memory)

        current_output = self.outputGate(controller_output)
        outputs.append(current_output)

        return current_output

    # If a controller is not passed in don't use this
    def forward(self, x):
        self.initialize_state()
        outputs = []

        for current_observation in x.transpose(0, 1):
            self.prev_reads.append(current_observation)
            controller_input = torch.cat(self.prev_reads, 1)
            controller_output = self.controller(controller_input)

            current_output = self.forward_step(controller_output)
            outputs.append(current_output)

        return torch.stack(outputs).transpose(0, 1)