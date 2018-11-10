import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np

from .Machine import Machine

class Controller(nn.Module):
    """
    Contains the two learnable parts of the model in four independent, fully connected layers.
    First the initial values for the registers and instruction registers and second the 
    parameters that computes the required distributions. 
    """

    def __init__(self, 
                 first_arg = None, 
                 second_arg = None, 
                 output = None, 
                 instruction = None, 
                 initial_registers = None,
                 ir = None,
                 stop_threshold = 0.9, 
                 multiplier = 5,
                 correctness_weight = .2, 
                 halting_weight = .2, 
                 confidence_weight = .2, 
                 efficiency_weight = .4,
                 diversity_weight = 0,
                 optimize = False,
                 mix_probabilities=False,
                 t_max = 75):
        """
        Initialize a bunch of constants and pass in matrices defining a program.
        
        :param first_arg: Matrix with the 1st register argument for each timestep stored in the columns (RxM)
        :param second_arg: Matrix with the 2nd register argument for each timestep stored in the columns (RxM)
        :param output: Matrix with the output register for each timestep stored in the columns (RxM)
        :param instruction: Matrix with the instruction for each timestep stored in the columns (NxM)
        :param initial_registers: Matrix where each row is a distribution over the value in one register (RxM)
        :param stop_threshold: The stop probability threshold at which the controller should stop running
        :param multiplier: The factor our vectors are be multiplied by before they're softmaxed to add blur
        :param correctness_weight: Weight given to the correctness component of the loss function
        :param halting_weight: Weight given to the halting component of the loss function
        :param confidence_weight: Weight given to the confidence component of the loss function
        :param efficiency_weight: Weight given to the efficiency component of the loss function
        :param optimize: Whether the ANC should optimize or not
        :param t_max: Maximum number of iterations of the program
        """
        super(Controller, self).__init__()
        
        # Initialize dimension constants
        R, M = initial_registers.size()
        self.M = M
        self.R = R
        self.times = []
        
        # Initialize loss function weights
        # In the ANC paper, these scalars are called, alpha, beta, gamma, and delta
        self.correctness_weight = correctness_weight
        self.halting_weight = halting_weight
        self.confidence_weight = confidence_weight
        self.efficiency_weight = efficiency_weight
        self.diversity_weight = diversity_weight
        
        # And yet more initialized constants... yeah, there are a bunch, I know.
        self.t_max = t_max
        self.stop_threshold = stop_threshold
        self.multiplier = multiplier
        self.mix_probabilities = mix_probabilities

        self.optimize = optimize
        
        if ir is None:
            IR = torch.zeros(M)
            IR[0] = 1
        else:
            IR = ir
        
        if optimize:
            # Initialize parameters.  These are the things that are going to be optimized. 
            self.first_arg = nn.Parameter(multiplier * first_arg)
            self.second_arg = nn.Parameter(multiplier * second_arg)
            self.output = nn.Parameter(multiplier * output)
            self.instruction = nn.Parameter(multiplier * instruction) 
            self.registers = nn.Parameter(multiplier * initial_registers)
            self.IR = nn.Parameter(multiplier * IR)
        else:
            self.first_arg = multiplier * first_arg
            self.second_arg = multiplier * second_arg
            self.output = multiplier * output
            self.instruction = multiplier * instruction
            self.registers = multiplier * initial_registers
            self.register_buffer('IR', multiplier * IR)
        
        self.register_buffer('initial_stop_probability', torch.zeros(1))
        
        # Machine initialization
        self.machine = Machine(M, R)
        self.softmax = nn.Softmax(0)
    
    def forward(self, input, forward_train):
        if forward_train:
            return self.forward_train(input)
        else:
            return self.forward_predict(self, input)
        
    def forward_wrapper(self, input, output_mem, output_mask):
        return (self.forward_train(input, (output_mem, output_mask)),)
    
    def forward_train(self, input, output):
        """
        Runs the controller on a certain input memory matrix. It returns the loss.
        
        :param initial_memory: The state of memory at the beginning of the program.
        :param output: A tuple (output_memory, output_mask): 
            output_memory: The desired state of memory at the end of the program.
            output_mask: The parts of the output memory that are relevant.
        
        :return: Returns the training loss.
        """
        
        initial_memory = input
        output_memory = output[0]
        output_mask = output[1]
        
        self.memory = Variable(initial_memory)
        self.output_memory = Variable(output_memory)
        self.output_mask = Variable(output_mask)
        self.stop_probability = Variable(self.initial_stop_probability)
        
        # Copy registers so we aren't using the values from the previous iteration. Also
        # make both registers and IR into a probability distribution.
        registers = nn.Softmax(1)(self.registers)
        IR = self.softmax(self.IR)
        
        if self.mix_probabilities:
            first_arg = self.softmax(self.first_arg)
            second_arg = self.softmax(self.second_arg)
            output = self.softmax(self.output)
            instruction = self.softmax(self.instruction)
        
        # loss initialization
        self.confidence = 0
        self.efficiency = 1
        self.halting = 0
        self.correctness = 0
        self.diversity = 0
        
        t = 0 
        
        # Run the program, one timestep at a time, until the program terminates or whe time out
        while t < self.t_max and float(self.stop_probability) < self.stop_threshold: 
            if self.mix_probabilities:
                a = torch.matmul(first_arg, IR)
                b = torch.matmul(second_arg, IR)
                o = torch.matmul(output, IR)
                e = torch.matmul(instruction, IR)
            else:
                a = self.softmax(torch.matmul(self.first_arg, IR))
                b = self.softmax(torch.matmul(self.second_arg, IR))
                o = self.softmax(torch.matmul(self.output, IR))
                e = self.softmax(torch.matmul(self.instruction, IR))
                        
            # Update memory, registers, and IR after machine operation
            self.old_stop_probability = self.stop_probability
            self.memory, registers, IR, new_stop_prob = self.machine(e, a, b, o, self.memory, registers, IR)
            self.stop_probability = self.stop_probability + (new_stop_prob * (1 - self.stop_probability))
            self.timestep_loss(t)
            t += 1
        
        self.final_loss(t)
        self.times.append(t)
        return self.total_loss()
    
    def forward_prediction(self, input):
        """
        Runs the controller on a certain input memory matrix. It returns the output memory matrix.
        
        :param initial_memory: The state of memory at the beginning of the program.
        
        :return: Returns the output memory matrix.
        """
        memory = input[0]
        # Program's initial memory
        self.memory = Variable(memory)
        self.stop_probability = 0
        
        # Copy registers so we aren't using the values from the previous iteration. Also
        # make both registers and IR into a probability distribution.
        registers = nn.Softmax(1)(self.registers)
        IR = self.softmax(self.IR)
        
        if self.mix_probabilities:
            first_arg = self.softmax(self.first_arg)
            second_arg = self.softmax(self.second_arg)
            output = self.softmax(self.output)
            instruction = self.softmax(self.instruction)
        
        t = 0 
        
        # Run the program, one timestep at a time, until the program terminates or whe time out
        while t < self.t_max and float(self.stop_probability) < self.stop_threshold: 
            if self.mix_probabilities:
                a = torch.matmul(first_arg, IR)
                b = torch.matmul(second_arg, IR)
                o = torch.matmul(output, IR)
                e = torch.matmul(instruction, IR)
            else:
                a = self.softmax(torch.matmul(self.first_arg, IR))
                b = self.softmax(torch.matmul(self.second_arg, IR))
                o = self.softmax(torch.matmul(self.output, IR))
                e = self.softmax(torch.matmul(self.instruction, IR))
                        
            # Update memory, registers, and IR after machine operation
            self.old_stop_probability = self.stop_probability
            self.memory, registers, IR, new_stop_prob = self.machine(e, a, b, o, self.memory, registers, IR) 
            
            self.stop_probability = self.stop_probability + new_stop_prob
            t += 1
        
        return self.memory, None
    
    def timestep_loss(self, t):
        # Confidence Loss 
        mem_diff = self.output_memory - self.memory
        correctness = torch.sum(self.output_mask * mem_diff * mem_diff)
        self.confidence = self.confidence + (self.stop_probability - self.old_stop_probability) * correctness
        # Efficiency Loss
        self.efficiency = self.efficiency + (1 - self.stop_probability)               
    
    def final_loss(self, t):
        # Correctness loss
        mem_diff = self.output_memory - self.memory
        self.correctness = torch.sum(self.output_mask * mem_diff * mem_diff)

        # Halting loss
        if t == self.t_max:
            self.halting = (1 - self.stop_probability)
            
         # Diversity loss
        self.diversity = self.softmax(self.instruction).prod(1).sum(0)

    def total_loss(self):
        """ compute four diferent loss functions and return a weighted average of the four measuring correctness, 
        halting, efficiency, and confidence"""
        return  (self.correctness*self.correctness_weight) + (self.confidence_weight*self.confidence) + (self.halting_weight*self.halting) + (self.efficiency_weight*self.efficiency) + (self.diversity_weight*self.diversity)



