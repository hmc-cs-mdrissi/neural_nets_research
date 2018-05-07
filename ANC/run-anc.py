import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import random
import matplotlib.pyplot as plt


from .util import one_hotify
from .util import anc_validation_criterion
from .DummyDataset import DummyDataset
from .IncTaskDataset import IncTaskDataset
from .Controller import Controller
from ..neural_nets_library import training



# # Addition task
# # Generate this by running the instructions here (but with the addition program file): https://github.com/aditya-khant/neural-assembly-compiler
# # Then get rid of the .cuda in each of the tensors since we (or at least I) don't have cuda
# init_registers = torch.IntTensor([6,2,0,1,0,0]) # Length R, should be RxM
# first_arg = torch.IntTensor([4,3,3,3,4,2,2,5]) # Length M, should be RxM
# second_arg = torch.IntTensor([5,5,0,5,5,1,4,5]) # Length M, should be RxM
# target = torch.IntTensor([4,3,5,3,4,5,5,5]) # Length M, should be RxM
# instruction = torch.IntTensor([8,8,10,5,2,10,9,0]) # Length M, should be NxM

#Increment task
init_registers = torch.IntTensor([6,0,0,0,0,0,0])
first_arg = torch.IntTensor([5,1,1,5,5,4,6])
second_arg = torch.IntTensor([6,0,6,3,6,2,6])
target = torch.IntTensor([1,6,3,6,5,6,6])
instruction = torch.IntTensor([8,10,2,9,2,10,0])

# init_registers = torch.IntTensor([0,0,6,0,0,0]) ### Note that the paper has an Instruction Register on top
# first_arg = torch.IntTensor([0,1,1,0,0,4,0]) ##
# second_arg = torch.IntTensor([0,2,0,1,0,3,0]) ###
# target = torch.IntTensor([1,5,1,5,0,5,5])
# instruction = torch.IntTensor([8,10,2,9,2,10,0])

# torch.Tensor{0, f, 6, 0, 0, f}
# torch.Tensor{0, 1, 1, 0, 0, 4, f},  -- first arguments
#    torch.Tensor{f, 2, f, 1, f, 3, f},  -- second arguments 
#    torch.Tensor{1, 5, 1, 5, 0, 5, 5},  -- target register !
#    torch.Tensor{8,10, 2, 9, 2,10, 0}   -- instruction to operate OK

# # Access task
# init_registers = torch.IntTensor([0,0,0])
# first_arg = torch.IntTensor([0,1,1,0,2])
# second_arg = torch.IntTensor([2,2,2,1,2])
# target = torch.IntTensor([1,1,1,2,2])
# instruction = torch.IntTensor([8,2,8,9,0])


# # Dummy Task
# init_registers = torch.IntTensor([1])
# first_arg = torch.IntTensor([0, 0])
# second_arg = torch.IntTensor([0, 0])
# target = torch.IntTensor([0, 0])
# instruction = torch.IntTensor([1, 0])

# # Dummy task - 0.5 stop prob
# init_registers = torch.IntTensor([0, 0])
# first_arg = torch.IntTensor([0, 1, 0])
# second_arg = torch.IntTensor([0, 1, 0])
# target = torch.IntTensor([0, 1, 0])
# instruction = torch.IntTensor([1, 9, 0])



# Get dimensions we'll need
M = first_arg.size()[0]
R = init_registers.size()[0]
N = 11

# Turn the given tensors into matrices of one-hot vectors.
init_registers = one_hotify(init_registers, M, 0)
first_arg = one_hotify(first_arg, R, 1)
second_arg = one_hotify(second_arg, R, 1)
target = one_hotify(target, R, 1)
instruction = one_hotify(instruction, N, 1)

# instruction[, ]


# for i in range(R):
#     first_arg[i, 6] = 1.0/6
#     second_arg[i, 0] = 1.0/6
#     second_arg[i, 2] = 1.0/6
#     second_arg[i, 4] = 1.0/6
#     second_arg[i, 6] = 1.0/6
# for j in range(M):
#     init_registers[1, j] = 1.0/7
#     init_registers[5, j] = 1.0/7
    
# print("IR", init_registers)
# print("FA", first_arg)
# print("SA", second_arg)
# print("INST", instruction)
    
num_examples = 500#7200

# M = 8 # Don't change this (as long as we're using the add-task)
# dataset = AddTaskDataset(M, num_examples)
# dataset = TrivialAddTaskDataset(M, num_examples)

M = 7 # Don't change this (as long as we're using the inc-task)
dataset = IncTaskDataset(M, 5, num_examples)

# M = 5
# dataset = AccessTaskDataset(M, num_examples)

# M = 5
# dataset = TrivialAccessTaskDataset(M, num_examples)

# M = 2
# dataset = DummyDataset(M, num_examples)


data_loader = data.DataLoader(dataset, batch_size = 1) # Don't change this batch size.  You have been warned.

# Initialize our controller
controller = Controller(first_arg = first_arg, 
                        second_arg = second_arg, 
                        output = target, 
                        instruction = instruction, 
                        initial_registers = init_registers, 
                        stop_threshold = .9, 
                        multiplier = 1,
                        correctness_weight = 1, 
                        halting_weight = 5, 
                        efficiency_weight = 0.01, 
                        confidence_weight = 0.1, 
                        optimize = True,
                        t_max = 50) 

# Learning rate is a tunable hyperparameter. The paper used 1 or 0.1.
optimizer = optim.Adam(controller.parameters(), lr = 0.1)
plot_every = 10

best_model, train_plot_losses, validation_plot_losses = training.train_model_anc(
    controller, 
    data_loader,  
    optimizer, 
    num_epochs = 1, 
    print_every = 10, 
    plot_every = plot_every, 
    deep_copy_desired = False, 
    validation_criterion = anc_validation_criterion, 
    batch_size = 1) # In the paper, they used batch sizes of 1 or 5
    
    #kangaroo


plt.plot([x * plot_every for x in range(len(train_plot_losses))], train_plot_losses)
plt.title("Training Loss")
plt.show()

averages = [sum(controller.times[i * 10:i * 10 + 10])/10.0 for i in range(int(len(controller.times) / 10))]
plt.plot(range(len(averages)), averages)
# plt.plot(range(len(controller.times)), controller.times)
plt.title("Timesteps")
plt.show()

plt.plot([x * plot_every for x in range(len(validation_plot_losses))], validation_plot_losses)
plt.title("Validation Loss")
plt.show()



#octopus