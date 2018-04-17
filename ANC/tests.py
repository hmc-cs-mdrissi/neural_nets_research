# IGNORE THIS... WORK IN PROGRESS
# self.ops = [ 
#     0 Stop(M),
#     1 Zero(M),
#     2 Increment(M),
#     3 Add(M),
#     4 Subtract(M),
#     5 Decrement(M),
#     6 Min(M),
#     7 Max(M),
#     8 Read(M),
#     9 Write(M),
#     10 Jump(M)
# ]
N = 11


# # Stop Test
# M = 4
# R = 3
# init_registers = torch.IntTensor([0,0,0])
# first_arg = torch.IntTensor([0,0,0,0])
# second_arg = torch.IntTensor([0,0,0,0])
# target = torch.IntTensor([0,0,0,0])
# instruction =  torch.IntTensor([3, 0, 0, 0]) # OK


# init_registers = one_hotify(init_registers, M, 0)
# first_arg = one_hotify(first_arg, R, 1)
# second_arg = one_hotify(second_arg, R, 1)
# target = one_hotify(target, R, 1)
# instruction = one_hotify(instruction, N, 1)
# instruction[0,1] = 0.5
# instruction[0,2] = 0.5
# instruction[2,1] = 0.5
# instruction[2,2] = 0.5

# memory = torch.IntTensor([0,0,0,0])
# memory = one_hotify(memory, M, 0)

# # What we expect: stops after 3 iterations; reg should have  [0.5, 0.25, 0.25]


# # Write test

# M = 2
# R = 3
# init_registers = torch.IntTensor([1,1,0])
# first_arg = torch.IntTensor([1,0])
# second_arg = torch.IntTensor([0,0])
# target = torch.IntTensor([0,0])
# instruction =  torch.IntTensor([0,0]) #

# init_registers = one_hotify(init_registers, M, 0)
# first_arg = one_hotify(first_arg, R, 1)
# second_arg = one_hotify(second_arg, R, 1)
# target = one_hotify(target, R, 1)
# instruction = one_hotify(instruction, N, 1)
# instruction[0,0] = 0.5
# instruction[9,0] = 0.5

# memory = torch.IntTensor([0,0])
# memory = one_hotify(memory, M, 0)

# # What we expect: stops after 2 iterations; index 1 of memory should have value (0:.5; 1:.5)


# # Read test

# M = 2
# R = 3
# init_registers = torch.IntTensor([0,0,0])
# first_arg = torch.IntTensor([0,0,0,0,0,0])
# second_arg = torch.IntTensor([0,0,0,0,0,0])
# target = torch.IntTensor([0,0,0,0,0,0])
# instruction =  torch.IntTensor([0,10,0,0,2,0])

# init_registers = one_hotify(init_registers, M, 0)
# first_arg = one_hotify(first_arg, R, 1)
# second_arg = one_hotify(second_arg, R, 1)
# target = one_hotify(target, R, 1)
# instruction = one_hotify(instruction, N, 1)
# instruction[0,0] = 0.5
# instruction[9,0] = 0.5

# memory = torch.IntTensor([0,0])
# memory = one_hotify(memory, M, 0)

# # What we expect: stops after 2 iterations; index 1 of memory should have value (0:.5; 1:.5)


# Normal ops test

# M = 5
# R = 3
# init_registers = torch.IntTensor([0,0,0])
# first_arg = torch.IntTensor([0,0,0,0,0])
# second_arg = torch.IntTensor([1,1,1,1,1])
# target = torch.IntTensor([0,0,0,0,0])
# instruction =  torch.IntTensor([1,3,4,7,0])

# init_registers = one_hotify(init_registers, M, 0)
# first_arg = one_hotify(first_arg, R, 1)
# second_arg = one_hotify(second_arg, R, 1)
# target = one_hotify(target, R, 1)
# instruction = one_hotify(instruction, N, 1)
# # zero, inc
# instruction[1,0] = 0.5
# instruction[2,0] = 0.5

# # add, dec
# instruction[3,1] = 0.5
# instruction[5,1] = 0.5

# # sub, min
# instruction[4,2] = 0.5
# instruction[6,2] = 0.5

# # max, write
# instruction[7,3] = 0.5
# instruction[9,3] = 0.5


# memory = torch.IntTensor([0,1,2,3,4])
# memory = one_hotify(memory, M, 0)

# # What we expect: stops after 2 iterations; index 1 of memory should have value (0:.5; 1:.5)





# # Initialize our controller
# controller = Controller(first_arg = first_arg, 
#                         second_arg = second_arg, 
#                         output = target, 
#                         instruction = instruction, 
#                         initial_registers = init_registers, 
#                         stop_threshold = .9, 
#                         multiplier = 50,
#                         correctness_weight = 10, 
#                         halting_weight = 0, 
#                         efficiency_weight = 1, 
#                         confidence_weight = 0, 
#                         t_max = 50) 




N=11


# # AddTest
# M = 3
# R = 4

# init_registers = torch.IntTensor([0,1,0,0])
# first_arg = torch.IntTensor([2,0,3])
# second_arg = torch.IntTensor([1,2,3])
# target = torch.IntTensor([2,3,3])
# instruction = torch.IntTensor([3,9,0])

# # # DecTest
# M = 3
# R = 3
# init_registers = torch.IntTensor([1,0,0])
# first_arg = torch.IntTensor([0,1,2])
# second_arg = torch.IntTensor([2,0,2])
# target = torch.IntTensor([0,2,2])
# instruction = torch.IntTensor([5,9,0])

# # IncTest
# M = 3
# R = 3
# init_registers = torch.IntTensor([0,0,0])
# first_arg = torch.IntTensor([1,0,2])
# second_arg = torch.IntTensor([2,1,2])
# target = torch.IntTensor([1,2,2])
# instruction = torch.IntTensor([2,9,0])

# # JezTest
# M = 3
# R = 3
# init_registers = torch.IntTensor([2,0,1,0])
# first_arg = torch.IntTensor([1,1,2,3])
# second_arg = torch.IntTensor([3,0,1,3])
# target = torch.IntTensor([1,3,3,3])
# instruction = torch.IntTensor([1,10,9,0])

# # MaxTest
# M = 3
# R = 4
# init_registers = torch.IntTensor([2,1,0,0])
# first_arg = torch.IntTensor([1,2,3])
# second_arg = torch.IntTensor([0,1,3])
# target = torch.IntTensor([1,3,3])
# instruction = torch.IntTensor([7,9,0])

# # MinTest
# M = 3
# R = 4
# init_registers = torch.IntTensor([1,2,0,0])
# first_arg = torch.IntTensor([1,2,3])
# second_arg = torch.IntTensor([0,1,3])
# target = torch.IntTensor([1,3,3])
# instruction = torch.IntTensor([6,9,0])

# ReadTest
# M=3
# R=3
# init_registers = torch.IntTensor([0,0,0])
# first_arg = torch.IntTensor([0,1,2])
# second_arg = torch.IntTensor([2,0,2])
# target = torch.IntTensor([0,2,2])
# instruction = torch.IntTensor([8,9,0])

# # SubTest
# M = 3
# R = 4
# init_registers = torch.IntTensor([1,2,0,0])
# first_arg = torch.IntTensor([1,2,3])
# second_arg = torch.IntTensor([0,1,3])
# target = torch.IntTensor([1,3,3])
# instruction = torch.IntTensor([4,9,0])

# # WriteTest
# M = 2
# R = 3
# init_registers = torch.IntTensor([1,0,0])
# first_arg = torch.IntTensor([1,2])
# second_arg = torch.IntTensor([0,2])
# target = torch.IntTensor([2,2])
# instruction = torch.IntTensor([9,0])

# # ZeroTest
# M = 3
# R = 3
# init_registers = torch.IntTensor([0,1,0])
# first_arg = torch.IntTensor([1,0,2])
# second_arg = torch.IntTensor([2,1,2])
# target = torch.IntTensor([1,2,2])
# instruction = torch.IntTensor([1,9,0])

# init_registers = one_hotify(init_registers, M, 0)
# first_arg = one_hotify(first_arg, R, 1)
# second_arg = one_hotify(second_arg, R, 1)
# target = one_hotify(target, R, 1)
# instruction = one_hotify(instruction, N, 1)
# initial_memory = torch.zeros(M,M)
# initial_memory[:, 2] = 1
# # initial_memory[0,0] = 0
# # initial_memory[0,2] = 1
                                    
                                    
# controller = Controller(first_arg = first_arg, 
#                         second_arg = second_arg, 
#                         output = target, 
#                         instruction = instruction, 
#                         initial_registers = init_registers, 
#                         stop_threshold = .9, 
#                         multiplier = 50,
#                         correctness_weight = 10, 
#                         halting_weight = 0, 
#                         efficiency_weight = 1, 
#                         confidence_weight = 0, 
#                         t_max = 5) 
# print(controller.forward_train(initial_memory, initial_memory, initial_memory))