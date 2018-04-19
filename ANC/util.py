import torch
import random

def one_hotify(vec, number_of_classes, dimension):
    """
    Turn a tensor of integers into a matrix of one-hot vectors.
    
    :param vec: The vector to be converted.
    :param number_of_classes: How many possible classes the one hot vectors encode.
    :param dimension: Which dimension stores the elements of vec.  If 0, they're stored in the rows.  If 1, the columns.
    
    :return A matrix of one-hot vectors, each row or column corresponding to one element of vec
    """
    num_vectors = vec.size()[0]
    binary_vec = torch.zeros(num_vectors, number_of_classes)
    for i in range(num_vectors):
        binary_vec[i][vec[i]] = 1
    if dimension == 1:
        binary_vec.t_()
    
    return binary_vec


def anc_validation_criterion(output, label):
    output_mem = output[0].data
    target_memory = label[0]
    target_mask = label[1]
    
    output2 = output_mem * target_mask #
    target_memory = target_memory * target_mask
    _, target_indices = torch.max(target_memory, 2) #
    _, output_indices = torch.max(output2, 2) #
    return 1 - torch.equal(output_indices, target_indices)



def getBest(vec):
    maxVal, index = torch.max(vec, 0)
    if maxVal.data[0] > cutoff:
        return index.data[0]

def bestRegister(vec):
    index = getBest(vec)
    if index is not None:
        return "R" + str(1 + index)
    return "??"
    
def bestInstruction(vec):
    ops = [ 
        "STOP",
        "ZERO",
        "INC",
        "ADD",
        "SUB",
        "DEC",
        "MIN",
        "MAX",
        "READ",
        "WRITE",
        "JEZ"
    ]
    index = getBest(vec)
    if index is not None:
        return ops[index]
    return "??"


def printProgram():   
    
    print("IR = " + str(getBest(controller.IR)))
    
    # Print registers
    for i in range(R):
        print("R" + str(i + 1) + " = " + str(getBest(controller.registers[i,:])))

    print()

    # Print the actual program
    for i in range (M):
        print(bestRegister(controller.output[:, i]) + " = " + 
              bestInstruction(controller.instruction[:, i]) + "(" +
              bestRegister(controller.first_arg[:, i]) + ", " +
              bestRegister(controller.second_arg[:, i]) + ")")




    
def compareOutput():
    # compare our output to theirs
    # we get one point for every matching number
    match_count = 0
    softmax = nn.Softmax(0)
    for i in range(R):
        if getBest(nn.Softmax(1)(controller.registers)[i,:]) == orig_register[i]:
            match_count += 1
    for i in range (M):
        if getBest(softmax(controller.output)[:, i]) == orig_output[i]:
            match_count += 1
        if getBest(softmax(controller.instruction)[:, i]) == orig_instruction[i]:
            match_count += 1
        if getBest(softmax(controller.first_arg)[:, i]) == orig_first[i]:
            match_count += 1
        if getBest(softmax(controller.second_arg)[:, i]) == orig_second[i]:
            match_count += 1
    if getBest(softmax(controller.IR)) == orig_ir:
        match_count += 1
    
    percent_orig = match_count / (len(orig_register) + len(orig_output) + 
                                           len(orig_instruction) + len(orig_first) + len(orig_second) + 1)
    # print("PERCENT MATCH", percent_orig)
    return percent_orig

    